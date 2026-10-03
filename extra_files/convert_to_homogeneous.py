import numpy as np

def convert_to_homogeneous(v):
    """
    Convert points to homogeneous coordinates (GENERAL VERSION).

    Input:
        v : ndarray of shape (D, N)
            D = 2  -> 2D points
            D = 3  -> 3D points
            (works for any D)

    Output:
        v_h : ndarray of shape (D+1, N)
              homogeneous coordinates
    """
    if v.ndim != 2:
        raise ValueError("Input must be a 2D array of shape (D, N)")

    ones = np.ones((1, v.shape[1]))
    v_h = np.vstack((v, ones))
    return v_h