#! /usr/bin/env python3

from sensor_msgs.msg import CompressedImage
from settings import settings
from utils import decode_image, hsv_mask
import cv2
import numpy as np
from node import Node
from std_msgs.msg import Bool


localization_node = Node("localization_node")
max_reached = False
history = list()


def region_of_interest(img, vertices, match_mask_color=255):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def msg2view(msg: CompressedImage):
    global max_reached

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
    
    # mask = hsv_mask(cropped_image, np.array([102, 195, 0]), np.array([172, 255, 255])) #robotics
    mask = hsv_mask(cropped_image, np.array([90, 109, 0]), np.array([183, 255, 129]))
    i = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largestContour)
        area = cv2.contourArea(largestContour)

        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # cv2.putText(
        #     image, f"{area}", (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2
        # )

        # history.append(area)

        if area >= settings.MAX_AREA:
            max_reached = True
        elif max_reached and area <= settings.MIN_AREA:
            print("I left the Node!")
            max_reached = False
            msg = Bool()
            msg.data = True
            localization_node.publish("localizer", msg)

    s = np.concatenate((image, i), axis=1)
    cv2.imshow("cv_img", s)
    cv2.waitKey(1)


def main():
    localization_node.init_publisher("localizer", "algo/localization", Bool)
    localization_node.init_subscriber(
        "camera",
        "/raspberry/data/image",
        CompressedImage,
        callback=msg2view,
        buff_size=2**28,
    )

    localization_node.spin()


main()
