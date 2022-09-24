"""
calculate finger state (straight, half-curved, fully curved)
Rewritten by Xingjian Leng on 21, Jul, 2022
Credit to:
"""
from typing import List

import numpy as np


def cos_similarity(a, b, thumb):
    """
    calculate finger state through cosine similarity
    thumb is treated differently from other fingers
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_angle = np.dot(a, b) / (a_norm * b_norm)
    if thumb:
        if cos_angle >= 0.8:
            return 0
        elif cos_angle >= 0.6:
            return 1
        else:
            return 2
    else:
        if cos_angle >= 0.7:
            return 0
        elif cos_angle >= 0:
            return 1
        else:
            return 2


def finger_classifier_cos(finger_coords):
    # for the thumb
    thumb_seg1 = finger_coords[:3] - finger_coords[3:6]
    thumb_seg2 = finger_coords[6:9] - finger_coords[9:12]
    rtn = [cos_similarity(thumb_seg1, thumb_seg2, True)]

    # for all remaining four fingers
    for i in range(4):
        offset = (12 if i < 3 else 15) + i * 9
        tip_offset = 54 + i * 3
        finger_seg1 = (
            finger_coords[offset : offset + 3] - finger_coords[offset + 3 : offset + 6]
        )
        finger_seg2 = (
            finger_coords[offset + 6 : offset + 9]
            - finger_coords[tip_offset : tip_offset + 3]
        )
        rtn.append(cos_similarity(finger_seg1, finger_seg2, False))
    return rtn


def finger_states_encoding(coordinates, with_head=True) -> List[List[int]]:
    rtn = np.zeros((359, 5), dtype=int)
    for i, frame in enumerate(coordinates):
        rtn[i] = finger_classifier_cos(frame[9 + 3 * with_head :])
    return rtn
