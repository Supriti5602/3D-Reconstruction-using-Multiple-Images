import numpy as np

def triangulate_3D_point_DLT(y1, y2, P1, P2):
    """
    Linear triangulation using DLT
    """
    u1, v1 = y1[0], y1[1]
    u2, v2 = y2[0], y2[1]

    M = np.zeros((4, 4))

    M[0] = u1 * P1[2] - P1[0]
    M[1] = v1 * P1[2] - P1[1]
    M[2] = u2 * P2[2] - P2[0]
    M[3] = v2 * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(M)
    X = Vt[-1]
    X = X / X[-1]

    return X
