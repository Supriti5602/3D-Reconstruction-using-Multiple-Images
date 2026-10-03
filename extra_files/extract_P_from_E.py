import numpy as np

def extract_P_from_E(E):
    """
    Extract the 4 possible camera matrices P2 from essential matrix E
    """
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    u3 = U[:, 2]

    R1 = U @ W  @ Vt
    R2 = U @ W.T @ Vt

    t  =  u3
    t_ = -u3

    P = np.zeros((4, 3, 4))
    P[0] = np.hstack((R1, t.reshape(3,1)))
    P[1] = np.hstack((R1, t_.reshape(3,1)))
    P[2] = np.hstack((R2, t.reshape(3,1)))
    P[3] = np.hstack((R2, t_.reshape(3,1)))

    return P
