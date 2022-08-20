"""
user interface for visualization
users can upload txt file and press Visualize button to see the prediction
NOTE: Each frame has 25/26 3D coordinate points (extra one could be the head coordinate)

Rewritten by Xingjian Leng on 20, Jul, 2022
Credit to:
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image, ImageTk

from utils import read_txt, hand_types
from finger_classifier import finger_classifier_cos
from palm_classifier import get_palm_vector, get_palm_center
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
    head_offset = 1 if with_head else 0
    points_raw = read_txt(txt_path.get(), with_head)
    left_hand = varHand.get() == hand_types[0]
    hand = 0 if left_hand else 1
    points = np.array(points_raw)
    select_label.config(text="")

    fingers = np.empty((0, 3))
    for frame in points:
        finger_frame = np.vstack(
            (
                frame[3 * head_offset : 3 * head_offset + 3].reshape(-1, 3),
                frame[9 + 3 * head_offset :].reshape(-1, 3),
            )
        )
        fingers = np.vstack((fingers, finger_frame))
    coord_mean = np.mean(fingers, axis=0)
    off_x = coord_mean[0]
    off_y = coord_mean[1]
    off_z = coord_mean[2]

    for frame_num, frame in enumerate(points):
        # classify states for fingers
        finger_predict = finger_classifier_cos(frame[9 + 3 * head_offset :])
        thumb_label.config(text=f"Thumb: {finger_label[finger_predict[0]]}")
        index_label.config(text=f"Index: {finger_label[finger_predict[1]]}")
        middle_label.config(text=f"Middle: {finger_label[finger_predict[2]]}")
        ring_label.config(text=f"Ring: {finger_label[finger_predict[3]]}")
        pinky_label.config(text=f"Pinky: {finger_label[finger_predict[4]]}")

        plt.cla()
        plt.close("all")
        ax = plt.axes(projection="3d")
        # set the dimension
        ax.set_xlim3d(-0.25, 0.25)
        ax.set_ylim3d(-0.25, 0.25)
        ax.set_zlim3d(-0.25, 0.25)

        # extract data
        x, y, z = extract_points(frame, with_head)
        # length for each list should be the same
        assert len(x) == len(y) == len(z)

        # if head data is included, show it in the visualization
        if with_head:
            ax.scatter(
                x[0] - off_x, y[0] - off_y, z[0] - off_z, c="black", marker="o", s=60
            )

        # plot each pivot on axes
        for i in range(head_offset, len(x)):
            ax.scatter(
                x[i] - off_x,
                y[i] - off_y,
                z[i] - off_z,
                c=indices_to_colour(i, head_offset),
                s=30,
                alpha=0.6,
            )

        # plot the skeleton of hands on axes (except from the root)
        for i in range(head_offset + 1, len(x) - 1):
            if i not in map(lambda p: p + head_offset, blue_pivots):
                ax.plot(
                    [x[i] - off_x, x[i + 1] - off_x],
                    [y[i] - off_y, y[i + 1] - off_y],
                    [z[i] - off_z, z[i + 1] - off_z],
                    c="black",
                )

        root_coord = (
            x[head_offset] - off_x,
            y[head_offset] - off_y,
            z[head_offset] - off_z,
        )
        # plot all the skeleton from the root (5 fingers)
        for i in map(lambda p: p + head_offset, inner_pivots):
            ax.plot(
                [root_coord[0], x[i] - off_x],
                [root_coord[1], y[i] - off_y],
                [root_coord[2], z[i] - off_z],
                c="black",
            )

        # plot the palm normal vector
        palm_vector = get_palm_vector(frame, hand)
        palm_center = get_palm_center(frame)
        ax.quiver(
            palm_center[0] - off_x,
            palm_center[1] - off_y,
            palm_center[2] - off_z,
            palm_vector[0],
            palm_vector[1],
            palm_vector[2],
            color="red",
        )

        # extract the image
        img = plt.gcf()
        canvas = FigureCanvasAgg(img)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_arr = np.asarray(buf)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_arr))

        # update image to show animations
        movieLabel.place(relx=0.08, rely=0.15)
        movieLabel.config(image=img_tk)
        movieLabel.update()


# the GUI main frame for visualization the hand model
root = Tk()
varHead = StringVar()
varHand = StringVar()
root.title("Visualization Tool")
root.geometry("800x500")
movieLabel = Label(root, width=512, height=400)

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

button_show = Button(root, text="Visualize", command=run_input)
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
