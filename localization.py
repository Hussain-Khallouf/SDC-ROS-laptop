#! /usr/bin/env python3

from sensor_msgs.msg import CompressedImage
from utils import decode_image
import cv2
import numpy as np
from node import Node


def region_of_interest(img, vertices, match_mask_color=255):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def msg2view(msg: CompressedImage):
    image = decode_image(msg.data)

    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height),
        (0, height / 2),
        (width, height / 2),
        (width, height),
    ]

    cropped_image = region_of_interest(
        image,
        np.array([region_of_interest_vertices], np.int32),
        match_mask_color=[255, 255, 255],
    )

    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    lower = np.array([91, 130, 96])
    uppeer = np.array([132, 255, 202])
    mask = cv2.inRange(hsv, lower, uppeer)
    i = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largestContour)
        area = cv2.contourArea(largestContour)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(
            image, f"{area}", (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2
        )

        if area > 1600:
            pass

    s = np.concatenate((image, i), axis=1)
    cv2.imshow("cv_img", s)
    cv2.waitKey(1)


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
