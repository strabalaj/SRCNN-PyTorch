# image_utils.py
import cv2
import numpy as np

def load_image_sanitize(path):
    """
    Load an image from path and convert it to a 3-channel BGR image.
    Handles grayscale, BGRA, and unusual channel counts safely.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Image with shape (H, W, 3), dtype=np.uint8.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found or unable to load: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    if img.shape[2] == 3:
        return img

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    if img.shape[2] != 3:
        print(f"Warning: Unexpected number of channels ({img.shape[2]}) in image, keeping first 3 channels.")
        img = img[:, :, :3]

    return img
