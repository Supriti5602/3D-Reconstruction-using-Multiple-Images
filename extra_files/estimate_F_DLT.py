import numpy as np

def estimate_F_DLT(x1s, x2s):
    u1 = x1s[0, :]
    v1 = x1s[1, :]
    u2 = x2s[0, :]
    v2 = x2s[1, :]

    M = np.vstack([
        u2*u1,
        u2*v1,
        u2,
        v2*u1,
        v2*v1,
        v2,
        u1,
        v1,
        np.ones_like(u1)
    ]).T

    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1, :]
    F_tilde = v.reshape(3, 3)

    return F_tilde, S, v, M
