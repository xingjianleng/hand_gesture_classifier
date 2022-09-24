"""
The utility file for read/write csv/txt files.
The encoding for output csv file is:
Line 1: gesture, wrist movements, hand_type,
Line 2 - Line n: each line is a frame for 25/26 coordinates of data points
"""
import numpy as np

import csv
from pathlib import Path
import re

# attributes
fingers_name = ("Thumb", "Index finger", "Middle finger", "Ring finger", "Pinky")
hand_types = ("left", "right")
gestures = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "thumb up",
    "pinky up",
    "fist",
    "resting",
    "ok",
    "clench",
    "expand",
    "zoom in",
    "zoom out",
    "pinch",
)
wrist_movements = (
    "no movement",
    "flip",
    "left",
    "right",
    "upward",
    "downward",
    "wrist flexion",
    "wrist extension",
)

# the mapping from attributes to their indices
gesture_map = {gesture: i for i, gesture in enumerate(gestures)}
wrist_movement_map = {
    wrist_movement: i for i, wrist_movement in enumerate(wrist_movements)
}
hand_map = {hand_type: i for i, hand_type in enumerate(hand_types)}


def read_txt(data_file_path, with_head=True):
    frames = []
    frame = []
    data_er_frame = 26 if with_head else 25
    with open(data_file_path, "r") as fr:
        for i, line in enumerate(fr):
            coordinates = re.findall(r"\(.*?\)", line)
            if coordinates:
                # if the coordinate exists, get the data
                frame.extend(
                    [
                        float(coordinate)
                        for coordinate in coordinates[0][1:-1].split(",")
                    ]
                )
            if (i + 1) % data_er_frame == 0:
                frames.append(frame)
                frame = []
    return frames


def write_csv(data_file, save_path, attributes, with_head=True):
    # map the string gestures, wrist movements and hand types to its integer representation
    gesture = gesture_map[attributes[0]]
    wrist_movement = wrist_movement_map[attributes[1]]
    hand_type = hand_map[attributes[2]]

    # extract the path
    data_file_path = Path(data_file).expanduser().absolute()
    save_path_obj = Path(save_path).expanduser().absolute()
    csv_save_path = Path(f"{str(save_path_obj)}/{data_file_path.stem}.csv")

    # the txt input
    data_input = read_txt(data_file_path, with_head)

    # write the data with the movement labels
    with open(csv_save_path, "w") as fw:
        fw.truncate(0)
        csv_writer = csv.writer(fw)
        # write movements to the csv file
        csv_writer.writerow((gesture, wrist_movement, hand_type))
        csv_writer.writerows(data_input)


def read_csv(csv_data):
    # return format: (coordinates, movements), all in numpy format
    coordinates = []
    with open(csv_data, "r") as fr:
        csv_reader = csv.reader(fr)
        for i, line in enumerate(csv_reader):
            if i == 0:
                movements = np.array([int(x) for x in line])
            else:
                coordinates.append([float(x) for x in line])
    return np.array(coordinates), movements


def extract_wrist_data(coordinates):
    extracted_frame = coordinates[:, 3:6]  # rootPos
    extracted_frame = np.hstack((extracted_frame, coordinates[:, 12:15]))  # Thumb 0
    extracted_frame = np.hstack((extracted_frame, coordinates[:, 51:54]))  # Pinky 0
    for index in (5, 8, 11, 14, 18):
        extracted_frame = np.hstack(
            (extracted_frame, coordinates[:, index: index + 3])
        )
    return extracted_frame
