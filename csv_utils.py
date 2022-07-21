import csv
from pathlib import Path
import re

# attributes
fingers_name = ("Thumb", "Index finger", "Middle finger", "Ring finger", "Pinky")
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


def read_txt(data_file_path, with_head=False):
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


def write_csv(data_file, save_path, movement):
    # map the string gestures and movements to its integer representation
    gesture = gesture_map[movement[0]]
    wrist_movement = wrist_movement_map[movement[1]]

    # extract the path
    data_file_path = Path(data_file).expanduser().absolute()
    save_path_obj = Path(save_path).expanduser().absolute()
    csv_save_path = Path(f"{str(save_path_obj)}/{data_file_path.stem}.csv")

    # the txt input
    data_input = read_txt(data_file_path)

    # write the data with the movement labels
    with open(csv_save_path, "w") as fw:
        fw.truncate(0)
        csv_writer = csv.writer(fw)
        # write movements to the csv file
        csv_writer.writerow((gesture, wrist_movement))
        csv_writer.writerows(data_input)
