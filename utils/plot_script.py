import math
import numpy as np
import matplotlib

matplotlib.use("Agg") # Run `matplotlib.use("Agg")` before importing pyplot.
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        return xz_plane

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection = "3d")
    
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([0, radius])
    fig.suptitle(title, fontsize = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        'red', 'blue', 'black', 'red', 'blue',
        'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
        'darkred', 'darkred', 'darkred', 'darkred', 'darkred'
    ]

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    drawn_lines = []
    drawn_cols = []

    def update(index):
        nonlocal drawn_lines, drawn_cols

        for artist in drawn_lines:
            artist.remove()
        for artist in drawn_cols:
            artist.remove()
        drawn_lines = []
        drawn_cols = []

        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        plane = plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1]
        )
        drawn_cols.append(plane)

        if index > 1:
            (ln,) = ax.plot3D(
                trajec[:index, 0] - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1] - trajec[index, 1],
                linewidth=1.0,
                color='blue'
            )
            drawn_lines.append(ln)

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            (ln,) = ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth = linewidth,
                color = color
            )
            drawn_lines.append(ln)

        return drawn_lines + drawn_cols

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_number,
        interval=1000 / fps,
        repeat=False,
        blit=False
    )

    writer = FFMpegFileWriter(fps=fps, codec="mpeg4")
    ani.save(save_path, writer=writer)
    plt.close(fig)
