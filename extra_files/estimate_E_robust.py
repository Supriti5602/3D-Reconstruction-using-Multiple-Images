import numpy as np
from estimate_F_DLT import estimate_F_DLT
from enforce_essential import enforce_essential
from compute_epipolar_errors import compute_epipolar_errors


def estimate_E_robust(x1, x2, eps):
    """
    RANSAC estimate of essential matrix using normalized correspondences x1 and x2
    and a normalized inlier threshold eps.

    Inputs:
        x1 : Normalized keypoints in image 1 (2xN or 3xN)
        x2 : Normalized keypoints in image 2 (2xN or 3xN)
        eps: Normalized inlier threshold (float)

    Returns:
        E      : 3x3 essential matrix
        inliers: Boolean array of shape (N,)
        errs   : Epipolar errors for all points
        iters  : Number of RANSAC iterations run
    """

    # --------------------------------------------------
    # Ensure homogeneous coordinates
    # --------------------------------------------------
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, x1.shape[1]))))
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, x2.shape[1]))))

    N = x1.shape[1]
    sample_size = 8
    max_iters = 10000

    best_E = None
    best_inliers = None
    best_errs = None

    iters = 0

    # --------------------------------------------------
    # RANSAC loop
    # --------------------------------------------------
    for it in range(max_iters):
        iters += 1

        # Random 8-point sample
        idx = np.random.choice(N, sample_size, replace=False)
        x1_s = x1[:, idx]
        x2_s = x2[:, idx]

        # Estimate essential matrix
        E_approx, _, _, _ = estimate_F_DLT(x1_s, x2_s)
        E_candidate = enforce_essential(E_approx)

        # Symmetric epipolar error
        e1 = compute_epipolar_errors(E_candidate,   x1, x2) ** 2
        e2 = compute_epipolar_errors(E_candidate.T, x2, x1) ** 2
        errs_candidate = 0.5 * (e1 + e2)

        # Inlier test
        inliers_candidate = errs_candidate < (eps ** 2)
        num_inliers = np.sum(inliers_candidate)

        # Keep best model
        if best_inliers is None or num_inliers > np.sum(best_inliers):
            best_E = E_candidate
            best_inliers = inliers_candidate
            best_errs = errs_candidate

        # Early stop if very good model
        if num_inliers > 0.8 * N:
            break
    E = best_E
    inliers = best_inliers
    errs = best_errs
    
    return E, inliers, errs, iters
