#! /usr/bin/env python3

from pickle import FALSE
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return

    # Make a copy of the original image.
    img = np.copy(img)

    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (img.shape[0], img.shape[1], 3),
        dtype=np.uint8,
    )

    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    # Return the modified image.
    return img


def msg2view(msg: CompressedImage):
    image = decode_image(msg.data)

    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height - 20),
        (200, height / 2),
        (width - 200, height / 2),
        (width, height - 20),
    ]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 58, 147])
    uppeer = np.array([32, 184, 255])
    mask = cv2.inRange(hsv, lower, uppeer)
    i = cv2.bitwise_and(image, image, mask=mask)

    gray_image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(
        cannyed_image, np.array([region_of_interest_vertices], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25,
    )

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y2 - y1) / (x2 - x1)
            print(m)

    line_image = np.copy(image)

    # if lines is None:
    #     # print(0)
    #     pass
    # else:
    #     # print(len(lines))
    #     line_image = draw_lines(image, lines, color=[255, 0, 0], thickness=3)

    image = region_of_interest(
        image,
        np.array([region_of_interest_vertices], np.int32),
        match_mask_color=[255, 255, 255],
    )

    row1 = np.concatenate((image, i), axis=1)
    row2 = np.concatenate(
        (cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR), line_image), axis=1
    )
    out = np.concatenate((row1, row2), axis=0)
    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imshow("cv_img", out)
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
