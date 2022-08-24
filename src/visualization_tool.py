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
import torch
import torch.nn.functional as F

from custom_thread import CustomThread
from data_animation import generate_animation
from finger_classifier import finger_states_encoding
from gesture_classifier import ConvolutionNetGesture, transformation_gesture
from movement_classifier import ConvolutionNetMovement, transformation_movement
from pathlib import Path
import re
from time import sleep
from utils import read_txt, hand_types, extract_wrist_data, gestures, wrist_movements
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


def prediction(net, data, transform=None):
    if transform is not None:
        transformed_data = transform(data)
        data = torch.reshape(transformed_data, (1, *transformed_data.shape))
    pred = net(data)
    return pred


def clean_temp():
    temp_path = Path("../.temp")
    for file in temp_path.iterdir():
        file.unlink()


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
    # show the program is analysing
    program_status.config(text="Program is analysing (wait for around 5 seconds)")
    program_status.update()

    with_head = varHead.get() == head_options[0]
    points_raw = read_txt(txt_path.get(), with_head)
    left_hand = varHand.get() == hand_types[0]
    hand = 0 if left_hand else 1
    points = np.array(points_raw)
    select_label.config(text="")

    # start a new thread for calculating labels and making predictions
    finger_state_thread = CustomThread(finger_states_encoding, points)
    finger_state_thread.start()
    wrist_data = extract_wrist_data(points)

    # movement prediction
    wrist_movement_thread = CustomThread(
        prediction, movement_classifier, wrist_data, transformation_movement
    )
    wrist_movement_thread.start()

    # gesture movement prediction
    finger_state_thread.join()  # wait for this thread
    finger_states = finger_state_thread.rtn
    gesture_state_thread = CustomThread(
        prediction, gesture_classifier, finger_states, transformation_gesture
    )
    gesture_state_thread.start()

    filename = Path(txt_path.get()).stem
    temp_video_path = Path(f"../.temp/temp_{filename}.mp4")
    if not temp_video_path.exists():
        # put animation on the main thread
        animation = generate_animation(points, with_head, hand)
        animation.save(temp_video_path)
    video_reader = imageio.get_reader(temp_video_path)

    gesture_state_thread.join()  # wait for this thread
    wrist_movement_thread.join()  # wait for this thread

    # update the gesture and movement prediction (use softmax)
    wrist_move = torch.argmax(F.softmax(wrist_movement_thread.rtn, dim=1))
    gesture_move = torch.argmax(F.softmax(gesture_state_thread.rtn, dim=1))
    gesture_label.config(text="movement:" + gestures[gesture_move])
    movement_label.config(text="gesture:" + wrist_movements[wrist_move])

    # change the label that the analysis terminates
    program_status.config(text="")
    program_status.update()

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
        sleep(1 / 60)  # make sure video is played at 60fps


# the GUI main frame for visualization the hand model
root = Tk()
varHead = StringVar()
varHand = StringVar()
root.title("Visualization Tool")
root.geometry("1024x768")
movieLabel = Label(root, width=640, height=480)

# detect .temp folder, create if not exist
Path("../.temp").mkdir(parents=True, exist_ok=True)

# load gesture classifier & wrist movement classifier
gesture_classifier = ConvolutionNetGesture()
gesture_classifier.load_state_dict(torch.load("../models/conv_gesture.pt"))
movement_classifier = ConvolutionNetMovement()
movement_classifier.load_state_dict(torch.load("../models/conv_movement.pt"))

button_upload_txt = Button(root, text="Choose txt file", command=upload_txt)
button_upload_txt.place(relx=0.05, rely=0.07)
txt_path = Entry(root)
txt_path.place(relx=0.21, rely=0.07)

button_clean_temp = Button(root, text="Clean temporary files", command=clean_temp)
button_clean_temp.place(relx=0.755, rely=0.9)

head_option_label = Label(root, text="With head data: ", font="Helvetica 12 bold")
head_option_label.place(relx=0.66, rely=0.045)
varHead.set(head_options[0])
head_mode = OptionMenu(root, varHead, *head_options)
head_mode.place(relx=0.755, rely=0.045)

hand_type_label = Label(root, text="Hand type: ", font="Helvetica 12 bold")
hand_type_label.place(relx=0.68, rely=0.1)
varHand.set(hand_types[0])
hand_type = OptionMenu(root, varHand, *hand_types)
hand_type.place(relx=0.755, rely=0.1)

button_show = Button(root, text="Analyse", command=run_input)
button_show.place(relx=0.75, rely=0.155)

select_label = Label(root, text="", font="Helvetica 12 bold")
select_label.place(relx=0.58, rely=0.03)

program_status = Label(root, text="", font="Helvetica 18 bold")
program_status.place(relx=0.1, rely=0.9)

# prediction at right side of the window
hand_state_label = Label(root, text="Hand State", font="Helvetica 14 bold")
hand_state_label.place(relx=0.75, rely=0.25)
gesture_label = Label(root, text="gesture:", font="Helvetica 12 bold")
gesture_label.place(relx=0.75, rely=0.3)
movement_label = Label(root, text="movement:", font="Helvetica 12 bold")
movement_label.place(relx=0.75, rely=0.35)

finger_state_label = Label(root, text="Finger State", font="Helvetica 14 bold")
finger_state_label.place(relx=0.75, rely=0.45)
thumb_label = Label(root, text="thumb:", font="Helvetica 12 bold")
thumb_label.place(relx=0.75, rely=0.5)
index_label = Label(root, text="index:", font="Helvetica 12 bold")
index_label.place(relx=0.75, rely=0.55)
middle_label = Label(root, text="middle:", font="Helvetica 12 bold")
middle_label.place(relx=0.75, rely=0.6)
ring_label = Label(root, text="ring:", font="Helvetica 12 bold")
ring_label.place(relx=0.75, rely=0.65)
pinky_label = Label(root, text="pinky:", font="Helvetica 12 bold")
pinky_label.place(relx=0.75, rely=0.7)

root.mainloop()
