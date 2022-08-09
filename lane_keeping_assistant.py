#! /usr/bin/env python3

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int8
from utils import decode_image
import cv2
import numpy as np
from node import Node

image_viewer = Node("image_viewer_node")


def msg2view(msg: CompressedImage):
    image = decode_image(msg.data)
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 58, 147])
    uppeer = np.array([32, 184, 255])
    mask = cv2.inRange(hsv, lower, uppeer)
    edge_image = cv2.adaptiveThreshold(
        mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )

    start_height = height - 50
    signed_thresh_From = int(width * 0.25)
    signed_thresh_To = int(width * 0.75)
    signed_thresh = edge_image[start_height].astype(np.int16)  # select only one row
    signed_thresh = signed_thresh[signed_thresh_From:signed_thresh_To]
    diff = np.diff(signed_thresh)  # The derivative of the start_height line
    points = np.where(
        np.logical_or(diff > 200, diff < -200)
    )  # maximums and minimums of derivative

    print(points[0].shape)
    # sleep(100)
    # print(f'len: {len(points)} {points}')

    cv2.line(
        image,
        (signed_thresh_From, start_height),
        (signed_thresh_To, start_height),
        (0, 255, 0),
        2,
    )  # draw horizontal line where scanning
    cv2.line(
        image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2
    )  # draw horizontal line where scanning

    if len(points) > 0 and len(points[0]) > 1:  # if finds something like a black line

        left = points[0][len(points[0]) - 4]
        right = points[0][len(points[0]) - 1]

        try:
            middle = ((left + signed_thresh_From) + (right + signed_thresh_From)) // 2
            # print((middle, start_height))
            cv2.circle(
                image, (left + signed_thresh_From, start_height), 2, (255, 0, 0), -1
            )
            cv2.circle(
                image, (right + signed_thresh_From, start_height), 2, (255, 0, 0), -1
            )
            cv2.circle(image, (middle, start_height), 2, (0, 0, 255), -1)

            result = Int8()

            if middle > width // 2:
                result.data = -2
            elif middle < width // 2:
                result.data = 2

            image_viewer.publish("angle", result)
        except:
            pass

    cv2.imshow("cv_img", image)
    cv2.waitKey(1)


def main():

    image_viewer.init_publisher("angle", "lane_keeping_assistant/angle", Int8)
    image_viewer.init_subscriber(
        "camera",
        "/raspberry/data/image",
        CompressedImage,
        callback=msg2view,
        buff_size=2**28,
    )
    image_viewer.spin()


main()
