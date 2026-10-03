import numpy as np

def normalize_image_points(x_h, K):
    """
    Normalize homogeneous image points using camera calibration matrix K.

    Inputs:
        x_h : (3, N) homogeneous pixel points
        K   : (3, 3) camera calibration matrix

    Output:
        x_n : (3, N) normalized image points
    """

    # Inverse of camera calibration matrix
    K_inv = np.linalg.inv(K)

    # Apply normalization
    x_n = K_inv @ x_h

    return x_n
