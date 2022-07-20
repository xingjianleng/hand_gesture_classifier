"""
user interface for visualization
users can upload txt file and press Visualize button to see the prediction
"""
from tkinter import *
from tkinter import filedialog
import re
from PIL import Image, ImageTk

# import handClassifier
import numpy as np

# import fingerClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from csv_utils import read_txt


def upload_txt():
    select_file = filedialog.askopenfilename()
    if re.match(r".*\.txt", select_file) is None:
        return

    txt_path.delete(0, END)
    txt_path.insert(0, select_file)


def run_input():
    if txt_path.get() == "":
        select_label.config(text="Please select the txt file!")
        return
    points_raw = read_txt(txt_path.get())
    points = np.array(points_raw)
    select_label.config(text="")
    count = 0
    for frame in points:
        """
        # gesture and movement classification
        if ((count >= 19) & (count < len(whole))):
            run = handClassifier.Model(whole_raw[count - 19:count + 1])
            movement = run.movementClassification()
            movement_label.config(text='movement:' + movement)
        elif ((count >= 2) & (count < len(whole))):
            run = handClassifier.Model(whole_raw[count - 2:count + 1])
            gesture = run.gestureClassification()
            gesture_label.config(text='gesture:' + gesture)
        else:
            gesture_label.config(text='gesture:')
            movement_label.config(text='movement:')
        # finger state
        run = fingerClassifier.Model(whole[count])
        finger_pre = run.fingerClassification()
        thumb_label.config(text='thumb:' + finger_pre[0])
        index_label.config(text='index:' + finger_pre[1])
        middle_label.config(text='middle:' + finger_pre[2])
        ring_label.config(text='ring:' + finger_pre[3])
        pinky_label.config(text='pinky:' + finger_pre[4])
        """

        plt.cla()
        plt.close("all")
        ax = plt.axes(projection="3d")
        # set the dimension
        ax.set_xlim3d(-0.35, 0.15)
        ax.set_ylim3d(0.8, 1.3)
        ax.set_zlim3d(-0.1, 0.7)

        # plot
        x = [frame[0]]
        y = [frame[1]]
        z = [frame[2]]
        for i in range(9, 60, 3):
            x.append(frame[i])
            y.append(frame[i + 1])
            z.append(frame[i + 2])
        # use blue to mark the thumb
        ax.scatter(x[:5], y[:5], z[:5], c="blue", marker="o", s=30)
        # use green to mark all other fingers
        ax.scatter(x[5:], y[5:], z[5:], c="green", marker="o", s=30)
        # the line indicating the thumb
        ax.plot(xs=x[0:5], ys=y[0:5], zs=z[0:5], color="orange")
        # lines indicating all other fingers
        ax.plot(
            xs=[x[0]] + x[5:8], ys=[y[0]] + y[5:8], zs=[z[0]] + z[5:8], color="r"
        )  # index finger
        ax.plot(
            xs=[x[0]] + x[8:11], ys=[y[0]] + y[8:11], zs=[z[0]] + z[8:11], color="r"
        )  # middle finger
        ax.plot(
            xs=[x[0]] + x[11:14], ys=[y[0]] + y[11:14], zs=[z[0]] + z[11:14], color="r"
        )  # ring finger
        ax.plot(
            xs=[x[0]] + x[14:], ys=[y[0]] + y[14:], zs=[z[0]] + z[14:], color="r"
        )  # pinky finger
        # extract the image
        img = plt.gcf()
        canvas = FigureCanvasAgg(img)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_arr = np.asarray(buf)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_arr))
        #
        movieLabel.place(relx=0.08, rely=0.15)
        movieLabel.config(image=img_tk)
        movieLabel.update()
        count += 1


# the GUI for visualization the hand model
root = Tk()
var = StringVar()
root.title("Visualization")
root.geometry("800x500")
movieLabel = Label(root, width=512, height=400)

but_upload_txt = Button(root, text="Upload txt file", command=upload_txt)
but_upload_txt.place(relx=0.05, rely=0.07, relwidth=0.12, relheight=0.04)
txt_path = Entry(root)
txt_path.place(relx=0.18, rely=0.07, relwidth=0.40, relheight=0.04)

but_show = Button(root, text="Visualize", command=run_input)
but_show.place(relx=0.755, rely=0.1, relwidth=0.18, relheight=0.04)

select_label = Label(root, text="", font="Helvetica 10 bold")
select_label.place(relx=0.6, rely=0.03)

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
