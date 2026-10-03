import numpy as np
from triangulate_3D_point_DLT import triangulate_3D_point_DLT


def cheirality_check(P1, P2_candidates, x1_in, x2_in):
    """
    Select the correct P2 using the cheirality condition

    Inputs:
        P1            : 3x4 camera matrix (camera 1)
        P2_candidates : 4x3x4 array of possible camera 2 matrices
        x1_in, x2_in  : 3xN normalized inlier points

    Returns:
        P2_best : 3x4 correct camera matrix
        X_best  : 4xN triangulated 3D points
        best_idx: index (0–3) of chosen P2
    """

    num_in_front = []
    all_3D_solutions = []

    # Try all 4 camera solutions
    for k in range(4):
        P2 = P2_candidates[k, :, :]

        in_front = 0
        X_list = []

        # Triangulate all inlier points
        for i in range(x1_in.shape[1]):
            X = triangulate_3D_point_DLT(
                x1_in[:, i],
                x2_in[:, i],
                P1,
                P2
            )
            X_list.append(X)

            # Project point into both cameras
            x1_cam = P1 @ X
            x2_cam = P2 @ X

            # Cheirality condition
            if x1_cam[2] > 0 and x2_cam[2] > 0:
                in_front += 1

        num_in_front.append(in_front)
        all_3D_solutions.append(np.array(X_list).T)  # 4xN

    # Choose best camera
    best_idx = np.argmax(num_in_front)

    P2_best = P2_candidates[best_idx, :, :]
    X_best  = all_3D_solutions[best_idx]

    return P2_best, X_best, best_idx
