import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
from palm_classifier import get_palm_center, get_palm_vector


# define the backend for matplotlib
matplotlib.use("Agg")

# colors
colors = ("yellow", "purple", "red", "green", "blue")
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
        return colors[0]
    elif index in map(lambda x: x + offset, purple_pivots):
        return colors[1]
    elif index in map(lambda x: x + offset, red_pivots):
        return colors[2]
    elif index in map(lambda x: x + offset, green_pivots):
        return colors[3]
    elif index in map(lambda x: x + offset, blue_pivots):
        return colors[4]
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
    return np.array(x), np.array(y), np.array(z)


def update_animation(
    frame_num: int,
    points: np.ndarray,
    with_head: bool,
    hand: int,
    off_x: float,
    off_y: float,
    off_z: float,
    scatters,
    lines,
    quiver,
):
    frame = points[frame_num]
    head_offset = int(with_head)
    x, y, z = extract_points(frame, with_head)
    assert len(x) == len(y) == len(z)

    if head_offset:
        scatters[0].set_offsets(np.array([x[0] - off_x, y[0] - off_y]))
        scatters[0].set_3d_properties(z[0] - off_z, "z")

    for i in range(head_offset, len(x)):
        scatters[i].set_offsets(np.array([x[i] - off_x, y[i] - off_y]))
        scatters[i].set_3d_properties(z[i : i + 1] - off_z, "z")

    # hand skeletons
    counter = 0
    for i in range(head_offset + 1, len(x) - 1):
        # skeleton plot without connecting to hand root
        if i not in map(lambda p: p + head_offset, blue_pivots):
            lines[counter].set_data_3d(
                [x[i] - off_x, x[i + 1] - off_x],
                [y[i] - off_y, y[i + 1] - off_y],
                [z[i] - off_z, z[i + 1] - off_z],
            )
            counter += 1
    root_coord = (
        x[head_offset] - off_x,
        y[head_offset] - off_y,
        z[head_offset] - off_z,
    )
    # plot all the skeleton from the root (5 fingers)
    for i in map(lambda p: p + head_offset, inner_pivots):
        lines[counter].set_data_3d(
            [root_coord[0], x[i] - off_x],
            [root_coord[1], y[i] - off_y],
            [root_coord[2], z[i] - off_z],
        )
        counter += 1
    assert counter == len(lines)  # should update each Line3D object

    # plot the palm vector
    palm_center = get_palm_center(frame, with_head)
    palm_vector = get_palm_vector(frame, hand, with_head) + palm_center
    quiver.set_data_3d(
        [palm_center[0] - off_x, palm_vector[0] - off_x],
        [palm_center[1] - off_y, palm_vector[1] - off_y],
        [palm_center[2] - off_z, palm_vector[2] - off_z],
    )

    # return Artist objects
    return *scatters, *lines, quiver


def generate_animation(points: np.ndarray, with_head: bool, hand: int):
    total_frames = len(points)
    head_offset = int(with_head)

    # calculate offset for centralize the hand visualization
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

    # use a dictionary
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # initialize artists
    init_frame = points[0]
    init_x, init_y, init_z = extract_points(init_frame, with_head=with_head)
    assert len(init_x) == len(init_y) == len(init_z)

    scatters = []
    if head_offset:
        scatters.append(
            ax.scatter(
                init_x[0] - off_x,
                init_y[0] - off_y,
                init_z[0] - off_z,
                c="black",
                marker="o",
                s=60,
            )
        )
    for i in range(head_offset, len(init_x)):
        color = indices_to_colour(i, offset=head_offset)
        scatters.append(
            ax.scatter(
                init_x[i] - off_x,
                init_y[i] - off_y,
                init_z[i] - off_z,
                c=color,
                s=30,
                alpha=0.6,
            )
        )

    # hand skeletons
    lines = []
    for i in range(head_offset + 1, len(init_x) - 1):
        # skeleton plot without connecting to hand root
        if i not in map(lambda p: p + head_offset, blue_pivots):
            lines.append(
                ax.plot(
                    [init_x[i] - off_x, init_x[i + 1] - off_x],
                    [init_y[i] - off_y, init_y[i + 1] - off_y],
                    [init_z[i] - off_z, init_z[i + 1] - off_z],
                    c="black",
                )[0]
            )
    root_coord = (
        init_x[head_offset] - off_x,
        init_y[head_offset] - off_y,
        init_z[head_offset] - off_z,
    )
    # plot all the skeleton from the root (5 fingers)
    for i in map(lambda p: p + head_offset, inner_pivots):
        lines.append(
            ax.plot(
                [root_coord[0], init_x[i] - off_x],
                [root_coord[1], init_y[i] - off_y],
                [root_coord[2], init_z[i] - off_z],
                c="black",
            )[0]
        )

    # plot the palm vector
    palm_center = get_palm_center(init_frame)
    palm_vector = get_palm_vector(init_frame, hand, with_head) + palm_center
    quiver = ax.plot(
        [palm_center[0] - off_x, palm_vector[0] - off_x],
        [palm_center[1] - off_y, palm_vector[1] - off_y],
        [palm_center[2] - off_z, palm_vector[2] - off_z],
        color="red",
    )[0]

    # set the dimension
    ax.set_xlim3d(-0.25, 0.25)
    ax.set_ylim3d(-0.25, 0.25)
    ax.set_zlim3d(-0.25, 0.25)

    ani = animation.FuncAnimation(
        fig,
        update_animation,
        total_frames,
        fargs=(points, with_head, hand, off_x, off_y, off_z, scatters, lines, quiver),
        blit=True,
        interval=1000 / 60,
    )

    return ani
