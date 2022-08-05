import numpy as np
import cv2 as cv


def decode_image(image):
    np_arr = np.fromstring(image, np.uint8)
    image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    return image_np