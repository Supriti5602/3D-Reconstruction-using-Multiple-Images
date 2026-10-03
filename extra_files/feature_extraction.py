import cv2
import numpy as np

def feature_extraction(img1, img2):
    """
    Feature extraction and matching using SIFT + FLANN
    EXACTLY following Computer Exercise 2, Task 2.1.

    Inputs:
        img1, img2 : RGB images (uint8)

    Outputs:
        x1, x2     : matched 2D points (2 x N)
        kp1, des1  : keypoints and descriptors of image 1
        kp2, des2  : keypoints and descriptors of image 2
        matches    : good matches after Lowe's ratio test
        all_matches: all raw matches before ratio test
    """

    # Convert RGB to grayscale
    rgb2gray = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Create SIFT detector (same parameters as exercise)
    sift = cv2.SIFT_create(
        contrastThreshold=0.02,
        edgeThreshold=10,
        nOctaveLayers=3
    )

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(rgb2gray(img1), None)
    kp2, des2 = sift.detectAndCompute(rgb2gray(img2), None)

    # k-NN matching using FLANN
    all_matches = cv2.FlannBasedMatcher().knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    matches = []
    for m, n in all_matches:
        if m.distance < 0.75 * n.distance:
            matches.append(m)

    # Extract matched keypoint indices
    matches_np = np.array([[m.queryIdx, m.trainIdx] for m in matches]).T

    # Convert keypoints to NumPy arrays
    f1 = np.array([kp.pt for kp in kp1]).T   # (2, N1)
    f2 = np.array([kp.pt for kp in kp2]).T   # (2, N2)

    # Matched 2D points
    x1 = f1[:, matches_np[0, :]]
    x2 = f2[:, matches_np[1, :]]

    return x1, x2, kp1, des1, kp2, des2, matches, all_matches
