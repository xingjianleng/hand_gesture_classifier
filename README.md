# Gesture-Data-Classifier

**Gesture-Data-Classifier** is an integrated tool with graphical user interface (GUI) designed for hand gesture data collected from Oculus Quest 2 headset. It can preprocess raw data files, and visualize the hand skeletons, palm directions and head position. It can also make predictions on the gesture, wrist movements and finger states.

## Dependencies

Gesture-Data-Classifier installation requires Python3 or newer. Additional libraries listed below are also required to run Gesture-Data-Classifier. They can be installed with `pip install -r requirements.txt`.

- [imageio](https://pypi.org/project/imageio/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [Pillow](https://pypi.org/project/Pillow/)
- [torch](https://pypi.org/project/torch/)
- [torchvision](https://pypi.org/project/torchvision/)

## Usage

Before running the GUI for the first time, the model files for gesture and wrist movement predictions should be downloaded. Model files are stored in Git LFS. The installation guide can be found on the [official website](https://git-lfs.github.com/). After installing Git LFS, run `git lfs install` for initialisation. Then, model files can be downloaded with `git lfs pull` in the project root directory.

To run the GUI, change the current working directory to `src` (`cd src`), and use ` python visualization_tool.py`. The window similar to below should appear.

![](https://user-images.githubusercontent.com/70084445/192096703-e183628c-6272-411e-851b-a497aca49893.jpg)

Then, choose the data file and corresponding hand type, and click on the **Analyse**. Wait for a few seconds to generate the visualization video and predictions.

![](https://user-images.githubusercontent.com/70084445/192096711-78cc3546-a0aa-418b-a3f0-d6ae02d16013.jpg)

After videos are produced, the GUI will show the hand joints, palm directions and head position visualizations. The hand and finger states predictions will be updated on the right-hand side of the GUI.

![](https://user-images.githubusercontent.com/70084445/192096712-6192d744-0abf-4681-9203-3ea1eb65c35d.jpg)

The generated video files are stored in the `.temp` folder in the project root directory. If there are many video files and taking up too much disk space, clicking on the **Clean temporary files** button can remove all temporary files.

![](https://user-images.githubusercontent.com/70084445/192096714-9f13ce95-5542-4ce7-9082-df7cee66cd47.jpg)

## Note

The input data file should have a fixed 359 frames. If the recorded data is too long, cropping should be applied before feeding the data into the tool.

