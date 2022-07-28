"""
user interface for labelling data
users can upload video and txt, and label the txt file with a gesture
label and a movement label. then csv file with labels will be generated

Rewritten by Xingjian Leng on 20, Jul, 2022
Credit to:
"""
import imageio
from PIL import Image, ImageTk
import re
from tkinter import Tk, Button, Label, Entry, StringVar, END, ttk, filedialog

from csv_utils import gestures, wrist_movements, write_csv


def upload_txt():
    # event handler for selecting txt file from user computer
    select_file = filedialog.askopenfilename()
    if re.match(r".*\.txt", select_file) is None:
        return

    txt_path.delete(0, END)
    txt_path.insert(0, select_file)


def upload_video():
    # event handler for selecting video file from user computer
    select_file = filedialog.askopenfilename()
    if re.match(r".*\.mp4", select_file) is None:
        return

    video_path.delete(0, END)
    video_path.insert(0, select_file)


def run_generate():
    # event handler for writing data into csv file format
    if label_gesture.get() == "" or label_movement.get() == "":
        finger_label.config(text="Please select labels for all fingers!")
        generate_msg.config(text="")
        return
    finger_label.config(text="")
    movement = [label_gesture.get(), label_movement.get()]
    write_csv(txt_path.get(), "../labelled_data/", movement)
    generate_msg.config(text="Generate successful!")


def run_input():
    # event handler for read in the video and txt files of gestures
    if txt_path.get() == "" or video_path.get() == "":
        select_label.config(text="Please select video and txt file!")
        generate_msg.config(text="")
        return
    select_label.config(text="")
    video_reader = imageio.get_reader(video_path.get())
    for im in video_reader:
        current_image = Image.fromarray(im).resize((450, 450))
        img_tk = ImageTk.PhotoImage(image=current_image)
        movieLabel.img_tk = img_tk
        movieLabel.config(image=img_tk)
        movieLabel.pack(padx=10, pady=100)
        movieLabel.update()


# main frame
root = Tk()
var = StringVar()
root.title("Gesture Labelling Tool")
root.geometry("800x700")
movieLabel = Label(root)
movieLabel.pack(padx=10, pady=10)

# upload and input button
button_upload_video = Button(root, text="Upload Video", command=upload_video)
button_upload_video.place(relx=0.05, rely=0.03)
video_path = Entry(root)
video_path.place(relx=0.2, rely=0.03, relwidth=0.6)

button_upload_txt = Button(root, text="Upload txt file", command=upload_txt)
button_upload_txt.place(relx=0.05, rely=0.1)
txt_path = Entry(root)
txt_path.place(relx=0.2, rely=0.1, relwidth=0.6)

button_input = Button(root, text="Input", command=run_input)
button_input.place(relx=0.855, rely=0.07)

select_label = Label(root, text="", font="Helvetica 10 bold")
select_label.place(relx=0.78, rely=0.03)

# select labels
text_gesture = Label(root, text="Hand Gesture")
text_gesture.place(relx=0.05, rely=0.8)
label_gesture = ttk.Combobox(root, state="readonly", values=gestures)
label_gesture.place(relx=0.05, rely=0.85, relwidth=0.2, relheight=0.04)

text_movement = Label(root, text="Wrist Movement")
text_movement.place(relx=0.3, rely=0.8)
label_movement = ttk.Combobox(root, state="readonly", values=wrist_movements)
label_movement.place(relx=0.3, rely=0.85, relwidth=0.2, relheight=0.04)

finger_label = Label(root, text="", font="Helvetica 10 bold")
finger_label.place(relx=0.55, rely=0.85)

button_generate = Button(root, text="Generate csv File", command=run_generate)
button_generate.place(relx=0.05, rely=0.92, relwidth=0.2)

generate_msg = Label(root, text="", font="Helvetica 14 bold")
generate_msg.place(relx=0.3, rely=0.925)

root.mainloop()
