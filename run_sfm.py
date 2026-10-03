import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from project_helpers import get_dataset_info
from helper_functions import *

def run_sfm(dataset_num):
    
    # STRUCTURE FROM MOTION ALGORITHM
    
    np.random.seed(1)

    # --------------------------------------------------
    # STEP 0: Load dataset information
    # --------------------------------------------------
    
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset_num)
    N = len(img_names)
    print("\nNumber of images:", N)
    
    
    R_rel = []              # Relative rotations R_(i,i+1)
    P_rel = []              # Relative camera matrices [R|t]
    X_rel = []              # Relative triangulated 3D points

    feat_im = []            # Features per image
    desc_im = []            # Descriptors per image
    matches_im = []         # Matches per image pair
    inliers_indices = []    # RANSAC inliers per image pair

    print("\n==================================================================")
    print("STEP 1: Relative orientations estimation for consecutive image pairs")
    print("====================================================================")
    
    for i in range(N - 1):
        print("\n")
        print(f"Image pair ({i}, {i+1})")

        # Load images
        img1 = cv2.imread(img_names[i])
        img2 = cv2.imread(img_names[i + 1])

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Feature extraction (SIFT)
        x1, x2, kp1, des1, kp2, des2, matches, all_matches = \
            feature_extraction(img1, img2)
        print("\nKeypoints:", len(kp1), len(kp2))
        print("\nMatches:", len(matches))


        if i == 0:
            feat_im.append(kp1)
            desc_im.append(des1)

        feat_im.append(kp2)
        desc_im.append(des2)
        matches_im.append(matches)

        # Convert matched points to homogeneous form
        print("\nBefore Homogeneous points:", x1.shape, x2.shape)
        x1_h = convert_to_homogeneous(x1)   # (3×N)
        x2_h = convert_to_homogeneous(x2)
        print("\nHomogeneous points:", x1_h.shape, x2_h.shape)
        
        # Normalize image points using K_inverse
        x1_n = normalize_image_points(x1_h, K)
        x2_n = normalize_image_points(x2_h, K)
        print("\nNormalized points:", x1_n.shape, x2_n.shape)
        
        # Estimate Essential Matrix using RANSAC
        eps = pixel_threshold / K[0, 0]  # normalized threshold

        E, inliers, errs, iters = estimate_E_robust(
            x1_n,
            x2_n,
            eps
        )
        
        num_inliers = np.sum(inliers)
        inliers_indices.append(inliers)
        
        print("Essential matrix estimated")
        print("      Inliers:", num_inliers, "/", x1_n.shape[1])
        print("      RANSAC iterations:", iters)

        # Extract relative pose (R, t) from E
        
        # Camera 1 is P1=[I 0]
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

        # 4 possible P2 from E
        P2_candidates = extract_P_from_E(E)

        # Use only inlier correspondences
        x1_in = x1_n[:, inliers]
        x2_in = x2_n[:, inliers]

        epipolar_vals = np.sum(x2_in * (E @ x1_in), axis=0)

        print("\nEpipolar constraint check (inliers only):")
        print("      Mean |x2^T E x1| :", np.mean(np.abs(epipolar_vals)))
        print("      Max  |x2^T E x1| :", np.max(np.abs(epipolar_vals)))
        
        # Cheirality check:
        #   - triangulate points
        #   - choose P2 with most points in front of cameras
        
        P2_best, X_best, best_idx = cheirality_check(
            P1,
            P2_candidates,
            x1_in,
            x2_in
        )

        # Extract rotation and translation
        R_12 = P2_best[:, :3]
        t_12 = P2_best[:, 3]

        # Store relative results
        R_rel.append(R_12)
        P_rel.append(P2_best)
        X_rel.append(X_best)
        
        print("Cheirality check done")
        print("      Selected P2 index:", best_idx+1)
        print("      R shape:", R_12.shape)
        print("      t shape:", t_12.shape)
        print("      X shape:", X_best.shape)

    print("\nStored data summary:")
    print("  Relative rotations stored :", len(R_rel))
    print("  Relative cameras stored   :", len(P_rel))
    print("  3D point sets stored      :", len(X_rel))
    print("  Feature sets stored       :", len(feat_im))
    print("  Match sets stored         :", len(matches_im))
    
    print("\n===================================")
    print("STEP 2: UPGRADE TO ABSOLUTE ROTATIONS")
    print("=====================================")

    R_abs = []                 # Absolute rotations
    R_abs.append(np.eye(3))    # R_0 = Identity

    print("\nR_abs[0] = Identity matrix")
    print(R_abs[0])

    # Chain relative rotations
    for k in range(len(R_rel)):
        R_next = R_rel[k] @ R_abs[k]
        R_abs.append(R_next)

        print(f"\nAbsolute Rotation R_abs[{k+1}] computed")
        print(R_next)

    print("\nAll absolute rotations computed.")
    
    print("ORTHOGONALITY CHECK")
    
    for i, R in enumerate(R_abs):
        error = np.linalg.norm(R @ R.T - np.eye(3))
        print(f"Camera {i}: ||R R_transpose - I|| = {error:.6e}")

    print("\n============================================================")
    print("STEP 3: Initial 3D reconstruction from the selected image pair")
    print("==============================================================")
    
    step3_a = int(init_pair[0])
    step3_b = int(init_pair[1])
    print(f"\nInit pair indices: ({step3_a}, {step3_b})")
    
    # Get stored keypoints + descriptors for the init pair
    step3_kpA = feat_im[step3_a]
    step3_kpB = feat_im[step3_b]
    step3_desA = desc_im[step3_a]
    step3_desB = desc_im[step3_b]
    
    print("Stored descriptor shapes:")
    print("  desA:", None if step3_desA is None else step3_desA.shape)
    print("  desB:", None if step3_desB is None else step3_desB.shape)
    
    # Match descriptors between init images
    step3_all_matches = cv2.FlannBasedMatcher().knnMatch(step3_desA, step3_desB, k=2)
    
    step3_matches = []
    for step3_m, step3_n in step3_all_matches:
        if step3_m.distance < 0.75 * step3_n.distance:
            step3_matches.append(step3_m)
            
    print("\nInit-pair matches after Lowe ratio:", len(step3_matches))
    
    # Convert matches to index arrays
    step3_matches_np = np.array([[m3.queryIdx, m3.trainIdx] for m3 in step3_matches]).T
    

    # Convert keypoints to coordinate arrays and pick matched pixels
    step3_fA = np.array([kp.pt for kp in step3_kpA]).T  # (2, NA)
    step3_fB = np.array([kp.pt for kp in step3_kpB]).T  # (2, NB)

    # Matched pixel points (2, M)
    step3_pixA = step3_fA[:, step3_matches_np[0, :]]
    step3_pixB = step3_fB[:, step3_matches_np[1, :]]

    print("\nMatched pixel point shapes:")
    print("  pixA:", step3_pixA.shape, " pixB:", step3_pixB.shape)
    
    # Save descriptors for the matched points in image A
    step3_descX_all = step3_desA[step3_matches_np[0, :], :].T
    print("Descriptors for candidate 3D points (descX_all) shape:", step3_descX_all.shape)
    
    # Convert matched pixels to homogeneous + normalize with K_inverse

    step3_hA = convert_to_homogeneous(step3_pixA)       # (3, M)
    step3_hB = convert_to_homogeneous(step3_pixB)       # (3, M)

    step3_normA = normalize_image_points(step3_hA, K)   # (3, M)
    step3_normB = normalize_image_points(step3_hB, K)   # (3, M)

    print("\nInit-pair normalized shapes:")
    print("  normA:", step3_normA.shape, " normB:", step3_normB.shape)
    
    print(pixel_threshold)
    
    step3_eps = pixel_threshold / K[0, 0]
    step3_E, step3_inliers, step3_errs, step3_iters = estimate_E_robust(step3_normA, step3_normB, step3_eps)

    print("\nInit-pair Essential matrix estimated")
    print("  RANSAC iterations:", step3_iters)
    print("  Inliers:", int(np.sum(step3_inliers)), "/", step3_normA.shape[1])
    
    # Epipolar constraint check on inliers: x2^T E x1 ~ 0
    
    step3_inA = step3_normA[:, step3_inliers]
    step3_inB = step3_normB[:, step3_inliers]

    step3_epi_vals = np.sum(step3_inB * (step3_E @ step3_inA), axis=0)

    print("\nInit-pair epipolar constraint check (inliers only):")
    print("  Mean |x2^T E x1| :", float(np.mean(np.abs(step3_epi_vals))))
    print("  Max  |x2^T E x1| :", float(np.max(np.abs(step3_epi_vals))))
    
    # Cheirality triangulation: choose correct P2 and triangulate 3D
    
    step3_P1 = np.hstack((np.eye(3), np.zeros((3, 1))))   # [I|0]
    step3_P2_cands = extract_P_from_E(step3_E)

    step3_P2_best, step3_X_rel, step3_best_idx = cheirality_check(
        step3_P1,
        step3_P2_cands,
        step3_inA,
        step3_inB
    )

    print("\nInit-pair cheirality done")
    print("  Selected P2 index:", step3_best_idx+1)
    print("  X_rel shape:", step3_X_rel.shape)
    print("First two 3D points (homogeneous):")
    print(step3_X_rel[:, :2])

    # Keep descriptors only for INLIER 3D points
    
    step3_descX_inliers = step3_descX_all[:, step3_inliers]
    print("descX_inliers shape:", step3_descX_inliers.shape)
    
    # Move 3D points to WORLD coordinates using absolute rotation of camera a
    
    step3_Ra = R_abs[step3_a]                  # absolute rotation of camera a
    step3_X_world3 = step3_Ra.T @ step3_X_rel[:3, :]   # (3, N-inliers)
    step3_X_world_h = convert_to_homogeneous(step3_X_world3)  # (4, N-inliers)

    print("\nWorld-frame initial 3D points:")
    print("  X_world3 shape:", step3_X_world3.shape)
    print("  X_worldh shape:", step3_X_world_h.shape)
    
    # --------------------------------------------------
    # Visualization of initial 3D points (Optional)
    # --------------------------------------------------
    plot_initial_rel_3D_pair = False  # setting False for no plotting

    if plot_initial_rel_3D_pair:
        print("\nPlotting initial 3D points from init_pair...")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(
        step3_X_world_h[0, :],
        step3_X_world_h[1, :],
        step3_X_world_h[2, :],
        '.',
        markersize=2
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Filtered 3D Points (Initial Pair)')

        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])

        plt.show()

    print("\n===================================")
    print("STEP 4: ROBUST TRANSLATION ESTIMATION")
    print("=====================================")
    
    T_abs = [None] * N   # absolute translations for all cameras
    # T_abs[0] = np.zeros(3)
    
    for i in range(0, N):

        print(f"\nEstimating translation for camera {i}")
        
        step4_kp_i  = feat_im[i]
        step4_des_i = desc_im[i]

        step4_all_matches = cv2.FlannBasedMatcher().knnMatch(
            step3_descX_inliers.T,   # (N3D, 128)
            step4_des_i,             # (Ni, 128)
            k=2
        )

        step4_matches = []
        for m, n in step4_all_matches:
            if m.distance < 0.75 * n.distance:
                step4_matches.append(m)

        print("3D-2D matches:", len(step4_matches))
        
        step4_matches_np = np.array(
            [[m.queryIdx, m.trainIdx] for m in step4_matches]
        ).T
        
        step4_f_i = np.array([kp.pt for kp in step4_kp_i]).T   # (2, Ni)

        step4_pix_i = step4_f_i[:, step4_matches_np[1, :]]    # (2, M)
        step4_X_h   = step3_X_world_h[:, step4_matches_np[0, :]]  # (4, M)
        
        print("Matched shapes:")
        print("  step4_X_h :", step4_X_h.shape)
        print("  step4_pix :", step4_pix_i.shape)
        
        step4_x_h = convert_to_homogeneous(step4_pix_i)   # (3, M)
        step4_x_n = normalize_image_points(step4_x_h, K)  # (3, M)
        
        print("  step4_x_n :", step4_x_n.shape)
        
        step4_R = R_abs[i]
        det_R = np.linalg.det(step4_R)
        
        print("det(R_abs[%d]) = %.6f" % (i, det_R))

        if abs(det_R - 1.0) > 1e-3:
            print("WARNING: Rotation determinant not close to +1")
        
        step4_thresh = 3 * pixel_threshold / K[0, 0]
        
        T_i = estimate_T_robust(
        step4_x_n,
        step4_X_h,
        step4_R,
        step4_thresh
        )
        
        T_abs[i] = T_i
        print("T shape:", T_i.shape)
        
        # SANITY CHECK: reprojection error
        X_cam = step4_R @ step4_X_h[:3, :] + T_i.reshape(3, 1)
        x_proj = X_cam[:2, :] / X_cam[2, :]

        pixel_err = np.sqrt(np.sum((step4_x_n[:2, :] - x_proj) ** 2, axis=0))

        print("Mean reprojection error:", np.mean(pixel_err))
        print("Median reprojection error:", np.median(pixel_err))
        
    for i in range(N):
        print(f"T_abs[{i}] = {T_abs[i]}")

    # --------------------------------------------------
    # STEP 5: Plot all cameras (optional)
    # --------------------------------------------------
    
    plot_all_camera = False # setting false for no plotting
    
    if plot_all_camera:
        P_all = []

        for i in range(N):
            P_i = np.hstack((R_abs[i], T_abs[i].reshape(3, 1)))
            P_all.append(P_i)
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for P in P_all:
            plot_camera(P, scale=0.2, ax=ax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses (World Coordinates)')
        ax.set_box_aspect([1, 1, 1])

        plt.show()

    print("\n=================================")
    print("STEP 6: TRIANGULATION FOR ALL PAIRS")
    print("===================================")
    
    X_all = []   # list to store 3D points for each image pair
    for i in range(N - 1):

        print(f"\nTriangulating points for image pair ({i}, {i+1})")

        # Get matched 2D points for image pair (i, i+1)
        
        matches = matches_im[i]   # shape (2, M)

        f1 = np.array([kp.pt for kp in feat_im[i]]).T
        f2 = np.array([kp.pt for kp in feat_im[i + 1]]).T
        
        # Extract matched indices
        idx1 = np.array([m.queryIdx for m in matches])
        idx2 = np.array([m.trainIdx for m in matches])

        # Matched points
        x1 = f1[:, idx1]   # (2, M)
        x2 = f2[:, idx2]   # (2, M)

        # Convert to homogeneous coordinates
        x1_h = convert_to_homogeneous(x1)   # (3, M)
        x2_h = convert_to_homogeneous(x2)   # (3, M)

        # Normalize image points using K_inverse
        x1_n = normalize_image_points(x1_h, K)
        x2_n = normalize_image_points(x2_h, K)

        # Re-estimate Essential matrix and keep inliers
        eps = pixel_threshold / K[0, 0]
        _, inliers, _, _ = estimate_E_robust(x1_n, x2_n, eps)

        x1_in = x1_n[:, inliers]
        x2_in = x2_n[:, inliers]

        print("Inliers used for triangulation:", x1_in.shape[1])

        # Build camera matrices P_i and P_{i+1}
        P1 = np.hstack((R_abs[i],   T_abs[i].reshape(3, 1)))
        P2 = np.hstack((R_abs[i+1], T_abs[i+1].reshape(3, 1)))

        # Triangulate 3D points (DLT)
        X_pair = []

        for k in range(x1_in.shape[1]):
            X = triangulate_3D_point_DLT(
                x1_in[:, k],
                x2_in[:, k],
                P1,
                P2
            )
            X_pair.append(X)

        X_pair = np.array(X_pair).T   # (4, N_points)

        print("Triangulated 3D points shape:", X_pair.shape)
        print("First 2 triangulated points:")
        print(X_pair[:, :2])

        # keep only the points that lie in front of both cameras
        X_front= keep_points_in_front(X_pair,P1,P2)
        
        # Remove far outliers
        X_good = filter_far_points(X_front)
        print("Points after distance filtering:", X_good.shape[1])
        X_all.append(X_good)

    print("Total 3D point sets stored:", len(X_all))
    

    print("FIRST VISUALIZATION: 3D POINTS + CAMERAS")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ALL 3D points (from all image pairs)
    for X_pair in X_all:
        if X_pair.shape[1] == 0:
            continue

        # Homogeneous -> Euclidean
        X = X_pair[:3, :] / X_pair[3, :]

        ax.scatter(
            X[0, :],
            X[1, :],
            X[2, :],
            s=1,
            c='blue',
            alpha=0.5
        )

    # Plot ALL cameras
    for i in range(len(R_abs)):
        P = np.hstack((R_abs[i], T_abs[i].reshape(3, 1)))
        plot_camera(P, scale=0.2, ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('First 3D Reconstruction (Points + Cameras)')

    ax.set_box_aspect([1, 1, 1])

    ax.grid(True)
    plt.tight_layout()
    ax.view_init(elev=20, azim=-60)
    plt.show()

    print("FINAL VISUALIZATION: CLEANER VIEW")

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    for X_pair in X_all:
        if X_pair.shape[1] == 0:
            continue

        X = X_pair[:3, :] / X_pair[3, :]

        num_plot = min(3000, X.shape[1])
        idx = np.random.choice(X.shape[1], size=num_plot, replace=False)

        ax2.scatter(
            X[0, idx],
            X[1, idx],
            X[2, idx],
            s=1,
            alpha=0.5
        )

    for i in range(len(R_abs)):
        P = np.hstack((R_abs[i], T_abs[i].reshape(3, 1)))
        plot_camera(P, scale=0.2, ax=ax2)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Final 3D Reconstruction (Clean View)')

    ax2.set_box_aspect([1, 1, 1])
    ax2.grid(True)

    ax2.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()
