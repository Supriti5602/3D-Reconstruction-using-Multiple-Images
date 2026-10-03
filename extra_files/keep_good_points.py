import numpy as np


def keep_points_in_front(X_pair, P1, P2):
    """
    Keep only 3D points that are in front of BOTH cameras.

    X_pair : (4, N) homogeneous 3D points
    P1, P2 : (3, 4) camera matrices

    Returns:
    X_front : (4, M) homogeneous points
    """

    X_front = []

    for i in range(X_pair.shape[1]):
        X = X_pair[:, i]      # one homogeneous point (4,)

        # Project into both cameras
        x1_cam = P1 @ X
        x2_cam = P2 @ X

        # Cheirality condition
        if x1_cam[2] > 0 and x2_cam[2] > 0:
            X_front.append(X)

    if len(X_front) == 0:
        return np.empty((4, 0))

    return np.array(X_front).T

def filter_far_points(X_front):
    """
    Remove 3D points far from the center of gravity.

    X_front : (4, N) homogeneous points

    Returns:
    X_good : (4, M) homogeneous points
    """

    if X_front.shape[1] == 0:
        return X_front

    # 1. Convert homogeneous → Euclidean (3, N)
    X = X_front[:3, :] / X_front[3, :]

    # 2. Center of gravity
    X_bar = np.mean(X, axis=1, keepdims=True)

    # 3. Distances from center
    distances = np.linalg.norm(X - X_bar, axis=0)

    # 4. 90th percentile
    d90 = np.percentile(distances, 90)

    # 5. Threshold
    threshold = 5 * d90

    # 6. Keep good points
    X_good = X[:, distances <= threshold]

    # 7. Back to homogeneous
    X_good_h = np.vstack((X_good, np.ones(X_good.shape[1])))

    return X_good_h

