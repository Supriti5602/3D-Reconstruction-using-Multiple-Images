import numpy as np
from estimate_translation_DLT import estimate_translation_DLT

def estimate_T_robust(xs, Xs, R, inlier_threshold):
    """
    Robust translation estimation using RANSAC
    Reprojection error is computed INLINE
    """

    max_inliers = 0
    T_best = None
    N = xs.shape[1]
    iterations = 10000

    for _ in range(iterations):

        # Sample 2 correspondences
        idx = np.random.choice(N, 2, replace=False)
        xs_sample = xs[:, idx]
        Xs_sample = Xs[:, idx]

        # Estimate translation using DLT
        T_candidate = estimate_translation_DLT(xs_sample, Xs_sample, R)
        if T_candidate is None:
            continue

        # INLINE reprojection error computation
        X_cam = R @ Xs[:3, :] + T_candidate.reshape(3, 1)
        x_proj = X_cam[:2, :] / X_cam[2, :]

        errors = np.sqrt(
            np.sum((xs[:2, :] - x_proj) ** 2, axis=0)
        )

        # Count inliers
        inliers = errors < inlier_threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            T_best = T_candidate

    # Final check
    if T_best is None:
        print("WARNING: No inliers found, returning zero translation")
        return np.zeros(3)

    # Final inlier percentage
    X_cam = R @ Xs[:3, :] + T_best.reshape(3, 1)
    x_proj = X_cam[:2, :] / X_cam[2, :]
    errors = np.sqrt(np.sum((xs[:2, :] - x_proj) ** 2, axis=0))
    percentage = 100 * np.sum(errors < inlier_threshold) / N

    print(f"Percentage of inliers in T_robust: {percentage:.2f}%")

    return T_best
