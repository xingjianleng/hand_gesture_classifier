"""
user interface for visualization
users can upload txt file and press Visualize button to see the prediction
NOTE: Each frame has 25/26 3D coordinate points (extra one could be the head coordinate)

Rewritten by Xingjian Leng on 20, Jul, 2022
Credit to:
"""
import imageio
import numpy as np
from PIL import Image, ImageTk

from custom_thread import CustomThread
from data_animation import generate_animation
from utils import read_txt, hand_types
from finger_classifier import finger_states_encoding
from pathlib import Path
import re
from tkinter import Tk, Button, Label, Entry, StringVar, OptionMenu, END, filedialog

# the option for visualization data
head_options = ("With head", "Without head")
# labels for finger states
finger_label = ("straight", "half curve", "curve")
# indices for inner pivots (should be directly connected to root)
inner_pivots = (1, 5, 8, 11, 14)
# indices for purple coloring pivots
purple_pivots = (1, 14)
# indices for red coloring pivots
red_pivots = (2, 5, 8, 11, 15)
# indices for green coloring pivots
green_pivots = (3, 6, 9, 12, 16)
# indices for blue coloring pivots (they are also fingertips indices)
blue_pivots = (4, 7, 10, 13, 17)


def finger_state_labeller(int_label: int) -> str:
    assert int_label in {0, 1, 2}
    if int_label == 0:
        return "straight"
    elif int_label == 1:
        return "half-curved"
    else:
        return "curved"


def indices_to_colour(index, offset):
    # get the color from the index and the head_offset
    if index == offset:
        return "yellow"
    elif index in map(lambda x: x + offset, purple_pivots):
        return "purple"
    elif index in map(lambda x: x + offset, red_pivots):
        return "red"
    elif index in map(lambda x: x + offset, green_pivots):
        return "green"
    elif index in map(lambda x: x + offset, blue_pivots):
        return "blue"
    else:
        raise ValueError("Invalid input index or offset!")


def extract_points(frame, with_head=False):
    # helper function to extract coordinate data from each frame
    x, y, z = [frame[0]], [frame[1]], [frame[2]]
    if with_head:
        x.append(frame[3])
        y.append(frame[4])
        z.append(frame[5])
    head_offset = 3 if with_head else 0
    for i in range(9 + head_offset, 60 + head_offset, 3):
        x.append(frame[i])
        y.append(frame[i + 1])
        z.append(frame[i + 2])
    return x, y, z


def upload_txt():
    # event handler for upload txt file
    select_file = filedialog.askopenfilename()
    if re.match(r".*\.txt", select_file) is None:
        return

    txt_path.delete(0, END)
    txt_path.insert(0, select_file)


def run_input():
    # event handler to visualize the animation of hand gestures
    if txt_path.get() == "":
        select_label.config(text="Please select the txt file!")
        return
    with_head = varHead.get() == head_options[0]
    points_raw = read_txt(txt_path.get(), with_head)
    left_hand = varHand.get() == hand_types[0]
    hand = 0 if left_hand else 1
    points = np.array(points_raw)
    select_label.config(text="")

    # start a new thread for calculating labels and making predictions
    finger_state_thread = CustomThread(finger_states_encoding, points)
    finger_state_thread.start()

    # put animation on the main thread
    animation = generate_animation(points, with_head, hand)
    temp_video_path = Path(
        "temp_video.mp4"
    )  # TODO: put temp video in .temp folder with video name
    animation.save(temp_video_path)
    video_reader = imageio.get_reader(temp_video_path)

    # TODO: add a note that the program is analysing

    finger_state_thread.join()  # wait for this thread
    for i, im in enumerate(video_reader):
        current_image = Image.fromarray(im)
        img_tk = ImageTk.PhotoImage(image=current_image)
        movieLabel.img_tk = img_tk
        movieLabel.config(image=img_tk)
        movieLabel.place(relx=0.05, rely=0.2)
        movieLabel.update()

        # update finger label
        thumb_label.config(
            text="thumb:" + finger_state_labeller(finger_state_thread.rtn[i][0])
        )
        index_label.config(
            text="index:" + finger_state_labeller(finger_state_thread.rtn[i][1])
        )
        middle_label.config(
            text="middle:" + finger_state_labeller(finger_state_thread.rtn[i][2])
        )
        ring_label.config(
            text="ring:" + finger_state_labeller(finger_state_thread.rtn[i][3])
        )
        pinky_label.config(
            text="pinky:" + finger_state_labeller(finger_state_thread.rtn[i][4])
        )


# the GUI main frame for visualization the hand model
root = Tk()
varHead = StringVar()
varHand = StringVar()
root.title("Visualization Tool")
root.geometry("1024x768")
movieLabel = Label(root, width=640, height=480)

# TODO: Path detect .temp folder, create if not exist
# TODO: Load gesture classifier & wrist movement classifier

button_upload_txt = Button(root, text="Upload txt file", command=upload_txt)
button_upload_txt.place(relx=0.05, rely=0.07)
txt_path = Entry(root)
txt_path.place(relx=0.21, rely=0.07)

varHead.set(head_options[0])
head_mode = OptionMenu(root, varHead, *head_options)
head_mode.place(relx=0.755, rely=0.045)

varHand.set(hand_types[0])
hand_type = OptionMenu(root, varHand, *hand_types)
hand_type.place(relx=0.78, rely=0.1)

button_show = Button(root, text="Analyse", command=run_input)
button_show.place(relx=0.755, rely=0.155)

select_label = Label(root, text="", font="Helvetica 10 bold")
select_label.place(relx=0.58, rely=0.03)

# prediction at right side of the window
hand_state_label = Label(root, text="Hand State", font="Helvetica 12 bold")
hand_state_label.place(relx=0.75, rely=0.25)
gesture_label = Label(root, text="gesture:", font="Helvetica 10 bold")
gesture_label.place(relx=0.75, rely=0.3)
movement_label = Label(root, text="movement:", font="Helvetica 10 bold")
movement_label.place(relx=0.75, rely=0.35)

finger_state_label = Label(root, text="Finger State", font="Helvetica 12 bold")
finger_state_label.place(relx=0.75, rely=0.45)
thumb_label = Label(root, text="thumb:", font="Helvetica 10 bold")
thumb_label.place(relx=0.75, rely=0.5)
index_label = Label(root, text="index:", font="Helvetica 10 bold")
index_label.place(relx=0.75, rely=0.55)
middle_label = Label(root, text="middle:", font="Helvetica 10 bold")
middle_label.place(relx=0.75, rely=0.6)
ring_label = Label(root, text="ring:", font="Helvetica 10 bold")
ring_label.place(relx=0.75, rely=0.65)
pinky_label = Label(root, text="pinky:", font="Helvetica 10 bold")
pinky_label.place(relx=0.75, rely=0.7)

root.mainloop()
