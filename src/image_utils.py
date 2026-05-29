import cv2
import numpy as np


def pil_to_numpy(image):

    rgb_image = image.convert("RGB") if image.mode != "RGB" else image

    return np.array(rgb_image)


def bgr_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )