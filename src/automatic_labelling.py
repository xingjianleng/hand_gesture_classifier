"""
The automatic labelling tool which will parse the name of data files.
The data files are named in the convention of "name_gesture_wrist.txt"
"""
from csv_utils import write_csv
from pathlib import Path

# attributes
gesture_map = {
    "one": "one",
    "two": "two",
    "three": "three",
    "four": "four",
    "five": "five",
    "thumb": "thumb up",
    "pinky": "pinky up",
    "fist": "fist",
    "rest": "resting",
    "ok": "ok",
    "clench": "clench",
    "expand": "expand",
    "zoomin": "zoom in",
    "zoomout": "zoom out",
    "pinch": "pinch",
}
wrist_movement_map = {
    "no": "no movement",
    "flip": "flip",
    "left": "left",
    "right": "right",
    "up": "upward",
    "down": "downward",
    "flexion": "wrist flexion",
    "extension": "wrist extension",
}


def extract_label(labels):
    attributes = labels.split("_")[:3]
    return gesture_map[attributes[1]], wrist_movement_map[attributes[2]], attributes[0]


if __name__ == "__main__":
    data_path = Path("../data").expanduser().absolute()
    save_path = Path("../labelled_data").expanduser().absolute()
    for file_name in data_path.rglob("*.txt"):
        # if the file doesn't exist, write it into csv format
        if not (save_path / file_name.with_suffix(".csv").name).exists():
            movement = extract_label(file_name.stem)
            write_csv(file_name, save_path, movement)
