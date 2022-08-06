#! /usr/bin/env python3

from sensor_msgs.msg import CompressedImage
from utils import decode_image
import cv2 as cv
import numpy as np
from node import Node


def msg2view(msg: CompressedImage):
    decoded_image =decode_image (msg.data)
    cv.imshow("cv_img", decoded_image)
    cv.waitKey(1)


def main():
    image_viewer = Node("image_viewer_node")

    image_viewer.init_subscriber(
        "/raspberry/data/image",
        CompressedImage,
        callback=msg2view,
        buff_size=2**28,
    )
    image_viewer.spin()


main()