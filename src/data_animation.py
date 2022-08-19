import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from palm_classifier import get_palm_center, get_palm_vector


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


def generate_animation_wo_blit(points: np.ndarray, head_offset: int, hand: int):
    pass
