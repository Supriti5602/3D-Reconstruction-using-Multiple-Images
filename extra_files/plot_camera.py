import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def pflat(X):
    """
    Projective flattening: divide by last row
    X: (D, N)
    """
    return X / X[-1, :]

def camera_center_and_axis(P):
    """
    Compute camera center and principal axis from camera matrix P
    """
    # Camera center = null space of P
    C = pflat(la.null_space(P))[:3]

    # Principal axis (view direction)
    axis = P[-1, :3]
    axis = axis / np.linalg.norm(axis)

    return C, axis

def plot_camera(P, scale=0.2, ax=None):
    """
    Plot camera center and viewing direction
    """
    if ax is None:
        ax = plt.axes(projection='3d')

    C, axis = camera_center_and_axis(P)

    ax.scatter(C[0], C[1], C[2], c='g', s=30)
    d = axis * scale
    ax.quiver(C[0], C[1], C[2], d[0], d[1], d[2], color='r')

    return ax
