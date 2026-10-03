import numpy as np
import cv2
import scipy.linalg as la
import matplotlib.pyplot as plt

def cheirality_check(P1, P2_candidates, x1_in, x2_in):
    num_in_front = []
    all_3D_solutions = []

    for k in range(4):
        P2 = P2_candidates[k, :, :]

        in_front = 0
        X_list = []

        for i in range(x1_in.shape[1]):
            X = triangulate_3D_point_DLT(
                x1_in[:, i],
                x2_in[:, i],
                P1,
                P2
            )
            X_list.append(X)

            x1_cam = P1 @ X
            x2_cam = P2 @ X

            if x1_cam[2] > 0 and x2_cam[2] > 0:
                in_front += 1

        num_in_front.append(in_front)
        all_3D_solutions.append(np.array(X_list).T)  # 4xN

    best_idx = np.argmax(num_in_front)

    P2_best = P2_candidates[best_idx, :, :]
    X_best  = all_3D_solutions[best_idx]

    return P2_best, X_best, best_idx

def compute_epipolar_errors(F, x1s, x2s):
    l = F @ x1s

    a = l[0, :]
    b = l[1, :]
    c = l[2, :]

    u = x2s[0, :]
    v = x2s[1, :]

    num = np.abs(a*u + b*v + c)
    den = np.sqrt(a*a + b*b)

    return num / den

def convert_to_homogeneous(v):
    if v.ndim != 2:
        raise ValueError("Input must be a 2D array of shape (D, N)")

    ones = np.ones((1, v.shape[1]))
    v_h = np.vstack((v, ones))
    return v_h

def enforce_essential(E_approx):
    U, S, Vt = np.linalg.svd(E_approx)

    if np.linalg.det(U @ Vt) < 0:
        Vt[-1, :] *= -1

    S_new = np.diag([1, 1, 0])
    E = U @ S_new @ Vt
    return E

def estimate_E_robust(x1, x2, eps):
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

    for it in range(max_iters):
        iters += 1

        idx = np.random.choice(N, sample_size, replace=False)
        x1_s = x1[:, idx]
        x2_s = x2[:, idx]

        E_approx, _, _, _ = estimate_F_DLT(x1_s, x2_s)
        E_candidate = enforce_essential(E_approx)

        e1 = compute_epipolar_errors(E_candidate,   x1, x2) ** 2
        e2 = compute_epipolar_errors(E_candidate.T, x2, x1) ** 2
        errs_candidate = 0.5 * (e1 + e2)

        inliers_candidate = errs_candidate < (eps ** 2)
        num_inliers = np.sum(inliers_candidate)

        if best_inliers is None or num_inliers > np.sum(best_inliers):
            best_E = E_candidate
            best_inliers = inliers_candidate
            best_errs = errs_candidate

        if num_inliers > 0.8 * N:
            break
    E = best_E
    inliers = best_inliers
    errs = best_errs
    
    return E, inliers, errs, iters

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

def estimate_T_robust(xs, Xs, R, inlier_threshold):
    max_inliers = 0
    T_best = None
    N = xs.shape[1]
    iterations = 10000

    for _ in range(iterations):

        idx = np.random.choice(N, 2, replace=False)
        xs_sample = xs[:, idx]
        Xs_sample = Xs[:, idx]

        T_candidate = estimate_translation_DLT(xs_sample, Xs_sample, R)
        if T_candidate is None:
            continue

        X_cam = R @ Xs[:3, :] + T_candidate.reshape(3, 1)
        x_proj = X_cam[:2, :] / X_cam[2, :]

        errors = np.sqrt(
            np.sum((xs[:2, :] - x_proj) ** 2, axis=0)
        )

        inliers = errors < inlier_threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            T_best = T_candidate

    if T_best is None:
        print("WARNING: No inliers found, returning zero translation")
        return np.zeros(3)

    X_cam = R @ Xs[:3, :] + T_best.reshape(3, 1)
    x_proj = X_cam[:2, :] / X_cam[2, :]
    errors = np.sqrt(np.sum((xs[:2, :] - x_proj) ** 2, axis=0))
    percentage = 100 * np.sum(errors < inlier_threshold) / N

    print(f"Percentage of inliers in T_robust: {percentage:.2f}%")

    return T_best

def skew(x):
    return np.array([
        [ 0,     -x[2],  x[1]],
        [ x[2],   0,    -x[0]],
        [-x[1],  x[0],   0   ]
    ])

def estimate_translation_DLT(xs, Xs, R):
    x1 = xs[:, 0]
    x2 = xs[:, 1]

    X1 = Xs[:, 0]
    X2 = Xs[:, 1]

    b1 = (skew(x1) @ R @ X1[:3]).reshape(3, 1)
    b2 = (skew(x2) @ R @ X2[:3]).reshape(3, 1)

    A1 = skew(x1)
    A2 = skew(x2)

    M = np.vstack([
        np.hstack([b1, A1]),
        np.hstack([b2, A2])
    ])

    _, _, Vt = np.linalg.svd(M)

    v = Vt[-1]

    T = v[1:4] / v[0]

    return T

def extract_P_from_E(E):
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

def feature_extraction(img1, img2):
    rgb2gray = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(
        contrastThreshold=0.02,
        edgeThreshold=10,
        nOctaveLayers=3
    )

    kp1, des1 = sift.detectAndCompute(rgb2gray(img1), None)
    kp2, des2 = sift.detectAndCompute(rgb2gray(img2), None)

    all_matches = cv2.FlannBasedMatcher().knnMatch(des1, des2, k=2)

    matches = []
    for m, n in all_matches:
        if m.distance < 0.75 * n.distance:
            matches.append(m)

    matches_np = np.array([[m.queryIdx, m.trainIdx] for m in matches]).T

    f1 = np.array([kp.pt for kp in kp1]).T
    f2 = np.array([kp.pt for kp in kp2]).T

    x1 = f1[:, matches_np[0, :]]
    x2 = f2[:, matches_np[1, :]]

    return x1, x2, kp1, des1, kp2, des2, matches, all_matches

def keep_points_in_front(X_pair, P1, P2):
    X_front = []

    for i in range(X_pair.shape[1]):
        X = X_pair[:, i]

        x1_cam = P1 @ X
        x2_cam = P2 @ X

        if x1_cam[2] > 0 and x2_cam[2] > 0:
            X_front.append(X)

    if len(X_front) == 0:
        return np.empty((4, 0))

    return np.array(X_front).T

def filter_far_points(X_front):
    if X_front.shape[1] == 0:
        return X_front

    X = X_front[:3, :] / X_front[3, :]

    X_bar = np.mean(X, axis=1, keepdims=True)

    distances = np.linalg.norm(X - X_bar, axis=0)

    d90 = np.percentile(distances, 90)

    threshold = 5 * d90

    X_good = X[:, distances <= threshold]

    X_good_h = np.vstack((X_good, np.ones(X_good.shape[1])))

    return X_good_h

def normalize_image_points(x_h, K):
    K_inv = np.linalg.inv(K)

    x_n = K_inv @ x_h

    return x_n

def pflat(X):
    return X / X[-1, :]

def camera_center_and_axis(P):
    C = pflat(la.null_space(P))[:3]

    axis = P[-1, :3]
    axis = axis / np.linalg.norm(axis)

    return C, axis

def plot_camera(P, scale=0.2, ax=None):
    if ax is None:
        ax = plt.axes(projection='3d')

    C, axis = camera_center_and_axis(P)

    ax.scatter(C[0], C[1], C[2], c='g', s=30)
    d = axis * scale
    ax.quiver(C[0], C[1], C[2], d[0], d[1], d[2], color='r')

    return ax

def triangulate_3D_point_DLT(y1, y2, P1, P2):
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
