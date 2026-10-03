import numpy as np

def skew(x):
    """
    Construct skew-symmetric matrix for a 3D vector x.
    x must be shape (3,)
    """
    return np.array([
        [ 0,     -x[2],  x[1]],
        [ x[2],   0,    -x[0]],
        [-x[1],  x[0],   0   ]
    ])

def estimate_translation_DLT(xs, Xs, R):
    """
    Estimate translation T using DLT with known rotation R.

    Inputs:
        xs : (3, N)  normalized 2D points in homogeneous coords
        Xs : (4, N)  3D points in homogeneous coords
        R  : (3, 3)  rotation matrix

    Output:
        T : (3,) translation vector
    """

    # --- take the first two correspondences (minimal set) ---
    x1 = xs[:, 0]          # (3,)
    x2 = xs[:, 1]          # (3,)

    X1 = Xs[:, 0]          # (4,)
    X2 = Xs[:, 1]          # (4,)

    # Force column vectors (3×1)
    b1 = (skew(x1) @ R @ X1[:3]).reshape(3, 1)
    b2 = (skew(x2) @ R @ X2[:3]).reshape(3, 1)

    A1 = skew(x1)
    A2 = skew(x2)

    # Build matrix M (6×4)
    M = np.vstack([
        np.hstack([b1, A1]),
        np.hstack([b2, A2])
    ])

    # --- solve M v = 0 using SVD ---
    _, _, Vt = np.linalg.svd(M)

    v = Vt[-1]                  # last row of V^T = last column of V

    # --- extract translation ---
    # MATLAB: T = V(2:end, end) ./ V(1,4)
    T = v[1:4] / v[0]

    return T
