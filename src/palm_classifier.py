"""
The classifier for detecting palm directions. The mechanism behind this is to use
the vector cross product of thumb vector and pinky vector. We use difference order
of cross product vectors to differentiate left and right hand
"""
import numpy as np

from enum import Enum


class Direction(Enum):
    FORWARD = "Forward"
    PARTIAL_FORWARD = "Partial forward"
    BACKWARD = "Backward"
    PARTIAL_BACKWARD = "Partial backward"
    OTHERS = "Others"


def get_palm_vector(frame, hand_type, with_head: bool = True):
    # 0 -> left hand, 1 -> right hand
    assert hand_type in {0, 1}
    index_vec = get_index_vec(frame, with_head)
    ring_vec = get_ring_vec(frame, with_head)
    if hand_type == 0:
        # left hand
        norm_vec = np.cross(ring_vec, index_vec)
    else:
        # right hand
        norm_vec = np.cross(index_vec, ring_vec)
    # 0.1 is the normalized length for the vector
    return 0.1 * norm_vec / np.linalg.norm(norm_vec)


def get_palm_center(frame, with_head: bool = True):
    # palm center is half of the length from root to the middle_finger_1
    offset = int(with_head) * 3
    root = frame[0 + offset: 3 + offset]
    middle_1 = frame[30 + offset: 33 + offset]
    return (middle_1 + root) / 2


def get_index_vec(frame, with_head: bool):
    # index vector is the vector point from root to index_finger_1
    offset = int(with_head) * 3
    index_1 = frame[21 + offset: 24 + offset]
    root = frame[0 + offset: 3 + offset]
    return index_1 - root


def get_ring_vec(frame, with_head: bool):
    # ring vector is the vector point from root to ring_finger_1
    offset = int(with_head) * 3
    ring_1 = frame[39 + offset: 42 + offset]
    root = frame[0 + offset: 3 + offset]
    return ring_1 - root


def get_head_vector(frame):
    # to call this function, the frame must contain head information
    palm_center = get_palm_center(frame=frame, with_head=True)
    head_pos = frame[0:3]
    return palm_center - head_pos


def forward_backward_classifier(palm_vector, head_vector):
    # cosine similarity between two vectors
    # 1 -> forward, 0 -> backward, -1 -> other cases
    cosine_sim = np.dot(palm_vector, head_vector) / (
            np.linalg.norm(palm_vector) * np.linalg.norm(head_vector)
    )
    if cosine_sim > np.sqrt(3) / 2:
        # within 30 degree range, forward case
        return Direction.FORWARD
    elif cosine_sim < -np.sqrt(3) / 2:
        # backward case
        return Direction.BACKWARD
    elif cosine_sim > np.sqrt(3) / 2:
        # within 60 degree range, partial forward
        return Direction.PARTIAL_FORWARD
    elif cosine_sim < -np.sqrt(3) / 2:
        # within 150 degree range, partial backward
        return Direction.PARTIAL_BACKWARD
    else:
        return Direction.OTHERS
