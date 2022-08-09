#! /usr/bin/env python3

from sensor_msgs.msg import CompressedImage
from utils import decode_image
import cv2 as cv
import numpy as np
from node import Node


##########drawing
# cv2.line(
#     image,
#     (signed_thresh_From, start_height),
#     (signed_thresh_To, start_height),
#     (0, 255, 0),
#     2,
# )  # draw horizontal line where scanning
# cv2.line(
#     image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2
# )  # draw horizontal line where scanning

# print((middle, start_height))
# cv2.circle(
#     image, (left + signed_thresh_From, start_height), 2, (255, 0, 0), -1
# )
# cv2.circle(
#     image, (right + signed_thresh_From, start_height), 2, (255, 0, 0), -1
# )
# cv2.circle(image, (middle, start_height), 2, (0, 0, 255), -1)

def raw_image(image):
    cv.imshow("cv_img", image)



def msg2view(msg: CompressedImage):
    decoded_image = decode_image(msg.data)
    raw_image(decoded_image)
    cv.waitKey(1)


def main():
    image_viewer = Node("image_viewer_node")

    image_viewer.init_subscriber(
        "camera",
        "/raspberry/data/image",
        CompressedImage,
        callback=msg2view,
        buff_size=2**28,
    )
    image_viewer.spin()


main()
