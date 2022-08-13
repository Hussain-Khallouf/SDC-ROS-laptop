#! /usr/bin/env python3

from math import fabs
from time import sleep
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int8
from utils import decode_image
from node import Node
from utils import hsv_mask
import cv2
import numpy as np

lane_keeping_assistant = Node("lane_keeping_assistant_node")


def msg2view(msg: CompressedImage):
    image = decode_image(msg.data)
    height, width = image.shape[:2]

    mask = hsv_mask(image, np.array([0, 58, 147]), np.array([32, 184, 255]))
    edge_image = cv2.adaptiveThreshold(
        mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )

    start_height = height - height // 2
    signed_thresh_From = int(width * 0)
    signed_thresh_To = int(width * 1)
    signed_thresh = edge_image[start_height].astype(np.int16)  # select only one row
    signed_thresh = signed_thresh[signed_thresh_From:signed_thresh_To]
    diff = np.diff(signed_thresh)  # The derivative of the start_height line
    points = np.where(
        np.logical_or(diff > 200, diff < -200)
    )  # maximums and minimums of derivative

    cv2.line(
        image,
        (signed_thresh_From, start_height),
        (signed_thresh_To, start_height),
        (0, 255, 0),
        2,
    )  # draw horizontal line where scanning
    cv2.line(image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

    if len(points) > 0 and len(points[0]) > 3:  # if finds something like a black line

        left = points[0][len(points[0]) - 4]
        right = points[0][len(points[0]) - 1]
        middle = ((left + signed_thresh_From) + (right + signed_thresh_From)) // 2

        cv2.circle(image, (left + signed_thresh_From, start_height), 2, (255, 0, 0), -1)
        cv2.circle(
            image, (right + signed_thresh_From, start_height), 2, (255, 0, 0), -1
        )
        cv2.circle(image, (middle, start_height), 2, (0, 0, 255), -1)

        result = Int8()
        # if fabs(middle - (width // 2)) > 50:
        if middle > width // 2:
            result.data = int((-fabs(middle - width / 2) / (width / 2)) * 100)
        elif middle < width // 2:
            result.data = int((fabs(middle - width / 2) / (width / 2)) * 100)
        print(result.data)
        lane_keeping_assistant.publish("angle", result)
            # sleep(0.2)

    cv2.imshow("cv_img", image)
    cv2.waitKey(1)


def main():

    lane_keeping_assistant.init_publisher("angle", "/algo/lane_keeping_assistant", Int8)
    lane_keeping_assistant.init_subscriber(
        "camera",
        "/raspberry/data/image",
        CompressedImage,
        callback=msg2view,
        buff_size=2**28,
    )
    lane_keeping_assistant.spin()


main()
