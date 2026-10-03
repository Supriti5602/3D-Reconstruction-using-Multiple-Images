import numpy as np

def enforce_essential(E_approx):
    U, S, Vt = np.linalg.svd(E_approx)

    if np.linalg.det(U @ Vt) < 0:
        Vt[-1, :] *= -1

    S_new = np.diag([1, 1, 0])
    E = U @ S_new @ Vt
    return E
