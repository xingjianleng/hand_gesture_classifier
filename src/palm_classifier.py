"""
The classifier for detecting palm directions. The mechanism behind this is to use
the vector cross product of thumb vector and pinky vector. We use difference order
of cross product vectors to differentiate left and right hand
"""
import numpy as np


def get_palm_vector(frame, hand_type):
    # 0 -> left hand, 1 -> right hand
    index_vec = get_index_vec(frame)
    ring_vec = get_ring_vec(frame)
    assert hand_type in {0, 1}
    if hand_type == 0:
        norm_vec = np.cross(ring_vec, index_vec)
    else:
        norm_vec = np.cross(index_vec, ring_vec)
    # 0.1 is the normalized length for the vector
    return 0.1 * norm_vec / np.linalg.norm(norm_vec)


def get_palm_center(frame):
    # palm center is half of the length from root to the middle_finger_1
    root = frame[3:6]
    middle_1 = frame[33:36]
    return (middle_1 + root) / 2


def get_index_vec(frame):
    # index vector is the vector point from root to index_finger_1
    index_1 = frame[24:27]
    root = frame[3:6]
    return index_1 - root


def get_ring_vec(frame):
    # ring vector is the vector point from root to ring_finger_1
    ring_1 = frame[42:45]
    root = frame[3:6]
    return ring_1 - root
