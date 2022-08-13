import numpy as np
import cv2 as cv


def decode_image(image):
    np_arr = np.fromstring(image, np.uint8)
    image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    return image_np


def hsv_mask(image, lower, upper):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    return mask
