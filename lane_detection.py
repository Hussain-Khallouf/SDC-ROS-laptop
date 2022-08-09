#! /usr/bin/env python3

from sensor_msgs.msg import CompressedImage
from utils import decode_image
import cv2
import numpy as np
from node import Node


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
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
        (0, height),
        (200, height / 2),
        (width - 200, height / 2),
        (width, height),
    ]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 0, 190])
    uppeer = np.array([255, 40, 255])
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
        minLineLength=50,
        maxLineGap=40,
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    line_image = image

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = image.shape[0] * (3 / 5)  # <-- Just below the horizon
    max_y = image.shape[0]  # <-- The bottom of the image

    if len(left_line_x) > 0 and len(left_line_y) > 0:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

    if len(right_line_x) > 0 and len(right_line_y) > 0:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

    line_image = image.copy()

    try:
        line_image = draw_lines(
            line_image,
            [
                [
                    [left_x_start, max_y, left_x_end, min_y],
                ]
            ],
            thickness=5,
        )
    except:
        # print('ERROR LEFT')
        pass

    try:
        line_image = draw_lines(
            line_image,
            [
                [
                    [right_x_start, max_y, right_x_end, min_y],
                ]
            ],
            thickness=5,
        )
    except:
        # print('ERROR RIGHT')
        pass

    # if lines is None:
    #     print(0)
    # else:
    #     print(len(lines))
    #     line_image = draw_lines(image, lines, color=[255, 0, 0], thickness=3)

    image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))

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
