"""
calculate finger state (straight, half-curved, fully curved)
Rewritten by Xingjian Leng on 21, Jul, 2022
Credit to:
"""
import numpy as np


# labels for finger states
label = ("straight", "half curve", "curve")


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
            return label[0]
        elif cos_angle >= 0.6:
            return label[1]
        else:
            return label[2]
    else:
        if cos_angle >= 0.7:
            return label[0]
        elif cos_angle >= 0:
            return label[1]
        else:
            return label[2]


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
