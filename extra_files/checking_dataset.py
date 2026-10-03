import os
import cv2
import numpy as np

from project_helpers import get_dataset_info


def main():
    # --- Choose which dataset to test ---
    dataset = 4   # change to 1..9

    # --- Get dataset info (K, image list, etc.) ---
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset)

    print("\n==============================")
    print("STEP 0: CHECK DATASET LOADING")
    print("==============================")
    print(f"Dataset number      : {dataset}")
    print(f"Number of images (N): {len(img_names)}")
    print(f"init_pair           : {init_pair}")
    print(f"pixel_threshold     : {pixel_threshold}")
    print("K =\n", K)

    # --- 1) Check that the returned values look reasonable ---
    if len(img_names) < 2:
        print("\nERROR: This dataset has fewer than 2 images. SfM needs at least 2.")
        return

    if not (isinstance(init_pair, (list, tuple)) and len(init_pair) == 2):
        print("\nERROR: init_pair should be a list/tuple of length 2.")
        return

    if not (0 <= init_pair[0] < len(img_names) and 0 <= init_pair[1] < len(img_names)):
        print("\nERROR: init_pair indices are out of range for this dataset.")
        return

    # --- 2) Check that all image paths exist ---
    missing = [p for p in img_names if not os.path.exists(p)]
    if missing:
        print("\nERROR: Some image files do not exist (wrong path or missing data folder):")
        for p in missing:
            print(" -", p)
        return

    print("\nAll image paths exist")

    # --- 3) Try to load each image and print its shape ---
    shapes = []
    for i, path in enumerate(img_names):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # loads as BGR
        if img is None:
            print(f"\nERROR: cv2.imread failed for image[{i}] = {path}")
            return

        h, w = img.shape[:2]
        shapes.append((h, w))
        print(f"image[{i}] loaded shape = (h={h}, w={w}, c={img.shape[2]})")

    # --- 4) Quick sanity check: sizes should match the values used to build K ---
    # (In your helper, K uses im_width and im_height from the dataset definition)
    print("\nSanity check:")
    print(" - First image shape:", shapes[0])
    print(" - All images same size?", all(s == shapes[0] for s in shapes))

    # --- 5) Display the first image (optional, but very helpful) ---
    img0 = cv2.imread(img_names[0], cv2.IMREAD_COLOR)
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # convert for correct colors if you later use matplotlib

    print("\nShowing the FIRST image in an OpenCV window.")
    print("Press any key on the image window to close it.")
    cv2.imshow("checking_dataset_0: first image", img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 6) Also display the init_pair images (optional but helpful) ---
    i1, i2 = init_pair
    imgA = cv2.imread(img_names[i1], cv2.IMREAD_COLOR)
    imgB = cv2.imread(img_names[i2], cv2.IMREAD_COLOR)

    cv2.imshow(f"init_pair image {i1}", imgA)
    cv2.imshow(f"init_pair image {i2}", imgB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nSTEP 0 DONE")
    print("Dataset info + K loaded correctly, and images are readable.\n")


if __name__ == "__main__":
    main()
