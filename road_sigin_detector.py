#! /usr/bin/env python3
import cv2
import numpy as np

from keras.models import load_model
import pandas as pd  
from node import Node
from sensor_msgs.msg import CompressedImage
from utils import decode_image
from settings import settings

road_sign_node = Node("road_sign_node")


def selecRed_And_Blue_Pixels(frame, lower_red, upper_red, lower_blue, upper_blue):

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Rmask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    Rmask2 = cv2.inRange(hsv, lower_red, upper_red)
    Rmask = cv2.bitwise_or(Rmask1, Rmask2)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask + Rmask)

    return res


def region_of_interest(img, vertices):
    rect = cv2.boundingRect(vertices)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()
    vertices = vertices - vertices.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [vertices], -1, (255, 255, 255), -1, cv2.LINE_AA)
    blackBack = cv2.bitwise_and(croped, croped, mask=mask)
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    whiteBack = bg + blackBack
    return whiteBack


def label_text(file):

    label_list = []
    r = pd.read_csv(file)

    for name in r["SignName"]:
        label_list.append(name)
    return label_list


def processImage(croped):

    blur = cv2.bilateralFilter(croped, 25, 100, 100)
    RGB = cv2.cvtColor(blur, cv2.COLOR_HSV2RGB)
    GRAY = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(GRAY, 10, 255, cv2.THRESH_BINARY)
    kernal = np.ones((10, 10), dtype=np.uint8)
    Morphological = cv2.dilate(thresh, kernal, iterations=1)
    canny = cv2.Canny(Morphological, 200, 255)

    return canny


def findContour(img):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def findBiggestContour(contours):
    c = [cv2.contourArea(i) for i in contours]
    if len(c) > 0:
        return contours[c.index(max(c))]
    else:
        return None


def boundaryBox(img, contours, sheft):

    x, y, w, h = cv2.boundingRect(contours)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    sign = img[y + sheft : (y + h - sheft), x + sheft : (x + w - sheft)]
    return img, sign.copy(), (x + (w // 2), y + (h // 2))


model = load_model("model-3x3.h5")
labels = label_text("label_names.csv")
offset = 5
height = settings.IMAGE_HEIGHT
width = settings.IMAGE_WIDTH
vertices = np.array(
    [[int(width * 0.5), 0], [width, 0], [width, height], [int(width * 0.5), height]]
)


def road_sign_viewer(msg: CompressedImage):
    imgOrignal = decode_image(msg.data)
    imgOrignal = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
    interestedImage = region_of_interest(imgOrignal, vertices=vertices)
    redSelcted = selecRed_And_Blue_Pixels(
        interestedImage,
        np.array([150, 50, 20]),
        np.array([170, 255, 255]),
        np.array([110, 50, 50]),
        np.array([130, 255, 255]),
    )

    canny = processImage(redSelcted)
    contours = findContour(canny)
    big = findBiggestContour(contours)
    img_With_Out_Line = interestedImage.copy()
    interestedImage_height = interestedImage.shape[0]
    interestedImage_width = interestedImage.shape[1]
    lineHeight = int(interestedImage_height * 0.4)
    cv2.line(
        interestedImage,
        (0, lineHeight),
        (interestedImage_width, lineHeight),
        (0, 0, 250),
        4,
    )

    if big is not None:
        contourArea = cv2.contourArea(big)
        if contourArea > 1500:
            interestedImage, sign, (x, y) = boundaryBox(img_With_Out_Line, big, 3)
            cv2.line(
                interestedImage,
                (0, lineHeight),
                (interestedImage_width, lineHeight),
                (255, 0, 0),
                4,
            )
            if (y < (lineHeight + offset)) or (y > (lineHeight - offset)):
                try:
                    sign = cv2.resize(sign, (32, 32))
                    to_predict = sign.reshape(1, 32, 32, 3)
                    scores = model.predict(to_predict)
                    cv2.imshow("sign", sign)
                    prediction = np.argmax(scores)
                    print("Label:", labels[prediction])
                    font = cv2.FONT_HERSHEY_DUPLEX
                    interestedImage = cv2.putText(
                        interestedImage,
                        labels[prediction] + "",
                        (0, 20),
                        font,
                        0.3,
                        (255, 255, 0),
                        1,
                    )
                except:
                    pass
    interestedImage = cv2.cvtColor(interestedImage, cv2.COLOR_RGB2BGR)
    cv2.imshow("Result", interestedImage)


def main():
    road_sign_node.init_subscriber()
    road_sign_node.init_subscriber(
        "camera",
        "/raspberry/data/image",
        CompressedImage,
        callback=road_sign_viewer,
        buff_size=2**28,
    )
    road_sign_node.spin()


main()
