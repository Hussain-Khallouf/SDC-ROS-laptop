#! /usr/bin/env python3


from settings import settings
from node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from utils import decode_image
import cv2
import numpy as np
import numpy.ma as ma


traffic_Light_node = Node("traffic_Light_node")
puplisher_name = "traffic_publisher"

def region_of_interest_Black_Background(img, vertices, is_white=False):
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(vertices)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()
    ## (2) make mask
    vertices = vertices - vertices.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [vertices], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    blackBack = cv2.bitwise_and(croped, croped, mask=mask)

    if is_white:
        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        blackBack = bg + blackBack

    return blackBack


def process_Image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    return gray, thresh


def findContour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def findBiggestContour(contours):
    c = [cv2.contourArea(i) for i in contours]
    if len(c) > 0:
        return contours[c.index(max(c))]
    else:
        return None


def boundaryBox(img, contours, sheft, color_Selected):
    x, y, w, h = cv2.boundingRect(contours)

    if color_Selected == "Move":
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    sign = img[y + sheft : (y + h - sheft), x + sheft : (x + w - sheft)]

    return img, sign.copy(), (x, y)


def selecBlue_AND_Red(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Rmask1 = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
    Rmask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    Rmask = cv2.bitwise_or(Rmask1, Rmask2)
    lower_blue = np.array([60, 100, 50])

    upper_blue = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=Rmask + mask)

    return res, Rmask + mask


def get_Color(traffic_Light):
    res, mask = selecBlue_AND_Red(traffic_Light)
    mask = cv2.merge((mask, mask, mask))
    mx = ma.masked_array(res, ~mask)
    average_color_row = np.average(mx, axis=0)
    average_color = np.average(average_color_row, axis=0)

    if np.argmax(average_color) == 0 or np.argmax(average_color) == 1:
        return "Move"
    else:
        return "Stop"


height = settings.IMAGE_HEIGHT
width = settings.IMAGE_WIDTH
vertices = np.array(
    [
        [int(width * 0.5), int(height * 0.25)],
        [int(width * 0.75), int(height * 0.25)],
        [width, int(height * 0.4)],
        [width, height],
        [int(width * 0.5), height],
    ]
)


def traffic_viewer(msg: CompressedImage):
    global vertices
    img_Original = decode_image(msg.data)

    interestedImage = region_of_interest_Black_Background(
        img_Original, vertices=vertices, is_white=True
    )
    img_With_Out_Line = interestedImage.copy()
    gray, thresh = process_Image(img_With_Out_Line)

    contours = findContour(thresh)

    big = findBiggestContour(contours)

    if big is not None:

        contourArea = cv2.contourArea(big)
        
        if contourArea > 3000:
            traffic_Light = region_of_interest_Black_Background(
                img_With_Out_Line, vertices=big
            )
            color_Selected = get_Color(traffic_Light)
            if color_Selected == "Move":
                msg = Bool()
                msg.data = True
                traffic_Light_node.publish(puplisher_name, msg)
                img_Original, sign, (x, y) = boundaryBox(
                    interestedImage, big, 0, "Move"
                )
                cv2.putText(
                    img=img_Original,
                    text=color_Selected,
                    org=(x + 10, y - 5),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.7,
                    color=(0, 255, 0),
                    thickness=1,
                )
            else:
                msg = Bool()
                msg.data = False
                traffic_Light_node.publish(puplisher_name, msg)
                img_Original, sign, (x, y) = boundaryBox(
                    interestedImage, big, 0, "Stop"
                )
                cv2.putText(
                    img=img_Original,
                    text=color_Selected,
                    org=(x + 10, y - 5),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),
                    thickness=1,
                )
            cv2.imshow("sign", traffic_Light)

    else: 
        msg = Bool()
        msg.data = True
        traffic_Light_node.publish(puplisher_name, msg)

    cv2.imshow("traffic", interestedImage)
    cv2.waitKey(1)


def main():
    traffic_Light_node.init_publisher(puplisher_name, "algo/traffic", Bool)
    traffic_Light_node.init_subscriber(
        "camera",
        "/raspberry/data/image",
        CompressedImage,
        callback=traffic_viewer,
        buff_size=2**28,
    )
    traffic_Light_node.spin()


main()
