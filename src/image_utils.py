import cv2
import numpy as np


def pil_to_numpy(image):

    return np.array(image)


def bgr_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )