import cv2

import numpy as np

from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def nothing(x):
    pass

def createRedTrackbar():

    # Create a black image, a window
    cv2.namedWindow("Red range")
    # create trackbars for color change

    cv2.createTrackbar("min_hue", "Red range", 170, 255, nothing)

    cv2.createTrackbar("max_hue", "Red range", 180, 255, nothing)

    cv2.createTrackbar("min_saturation", "Red range", 50, 255, nothing)

    cv2.createTrackbar("max_saturation", "Red range", 255, 255, nothing)

    cv2.createTrackbar("min_lightness", "Red range", 20, 255, nothing)

    cv2.createTrackbar("max_lightness", "Red range", 255, 255, nothing)

# def createBlueTrackbar():

#     # Create a black image, a window
#     cv2.namedWindow("Blue range")
#     # create trackbars for color change

#     cv2.createTrackbar("min_hue", "Blue range", 115, 255, nothing)

#     cv2.createTrackbar("max_hue", "Blue range", 125, 255, nothing)

#     cv2.createTrackbar("min_saturation", "Blue range", 100, 255, nothing)

#     cv2.createTrackbar("max_saturation", "Blue range", 255, 255, nothing)

#     cv2.createTrackbar("min_lightness", "Blue range", 25, 255, nothing)

#     cv2.createTrackbar("max_lightness", "Blue range", 255, 255, nothing)

#     # get current positions of four trackbars

def getRed_Range():

    min_hue = cv2.getTrackbarPos("min_hue", "Red range")

    max_hue = cv2.getTrackbarPos("max_hue", "Red range")

    min_saturation = cv2.getTrackbarPos("min_saturation", "Red range")

    max_saturation = cv2.getTrackbarPos("max_saturation", "Red range")

    min_lightness = cv2.getTrackbarPos("min_lightness", "Red range")

    max_lightness = cv2.getTrackbarPos("max_lightness", "Red range")

    lower_red = np.array([min_hue,min_saturation,min_lightness])

    upper_red = np.array([max_hue,max_saturation,max_lightness])


    return lower_red , upper_red

# def getBlue_Range():

#     min_hue = cv2.getTrackbarPos("min_hue", "Blue range")

#     max_hue = cv2.getTrackbarPos("max_hue", "Blue range")

#     min_saturation = cv2.getTrackbarPos("min_saturation", "Blue range")

#     max_saturation = cv2.getTrackbarPos("max_saturation", "Blue range")

#     min_lightness = cv2.getTrackbarPos("min_lightness", "Blue range")

#     max_lightness = cv2.getTrackbarPos("max_lightness", "Blue range")

#     lower_Blue = np.array([min_hue,min_saturation,min_lightness])

#     upper_Blue = np.array([max_hue,max_saturation,max_lightness])

#     return lower_Blue , upper_Blue

def selecRed_And_Blue_Pixels(frame , lower_red , upper_red , lower_blue , upper_blue):

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Rmask1 = cv2.inRange(hsv, (0,50,50), (10,255,255))

    Rmask2 = cv2.inRange(hsv, lower_red , upper_red)

    Rmask = cv2.bitwise_or(Rmask1, Rmask2 )

    red = cv2.bitwise_and(frame, frame, mask=Rmask)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue = cv2.bitwise_and(frame,frame, mask= mask )

    res = cv2.bitwise_and(frame,frame,mask = mask + Rmask )

    return res

def region_of_interest(img, vertices):


    ## (1) Crop the bounding rect


    rect = cv2.boundingRect(vertices)


    x,y,w,h = rect


    croped = img[y:y+h, x:x+w].copy()


    ## (2) make mask


    vertices = vertices - vertices.min(axis=0)


    mask = np.zeros(croped.shape[:2], np.uint8)


    cv2.drawContours(mask, [vertices], -1, (255, 255, 255), -1, cv2.LINE_AA)


    ## (3) do bit-op


    blackBack = cv2.bitwise_and(croped, croped, mask=mask)


    ## (4) add the white background


    bg = np.ones_like(croped, np.uint8)*255


    cv2.bitwise_not(bg,bg, mask=mask)


    whiteBack = bg+ blackBack


    return whiteBack

# Defining function for getting texts for every class - labels

def label_text(file):

    # Defining list for saving label in order from 0 to 42

    label_list = []


    # Reading 'csv' file and getting image's labels

    r = pd.read_csv(file)


    # Going through all names

    for name in r['SignName']:

        # Adding from every row second column with name of the label

        label_list.append(name)


    # Returning resulted list with labels


    return label_list

def processImage(croped):

    blur = cv2.bilateralFilter(croped, 25 , 100 , 100)


    RGB = cv2.cvtColor(blur, cv2.COLOR_HSV2RGB)


    GRAY = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)


    # #     منشان تجيب الحواف فانت لازم تجيب الخلفية بالابيض و الاسود


    _, thresh = cv2.threshold(GRAY, 10, 255, cv2.THRESH_BINARY)


    kernal = np.ones((10,10) , dtype=np.uint8)


    Morphological = cv2.dilate(thresh, kernal, iterations=1)


    # Morphological = cv2.erode(thresh, kernal , iterations = 1)


    canny = cv2.Canny(Morphological , 200 , 255)


    return blur , RGB , GRAY , thresh , Morphological , canny

def findContour(img):


	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	return contours

def findBiggestContour(contours):

    c = [cv2.contourArea(i) for i in contours]


    if len(c) > 0:

        return contours[c.index(max(c))]


    else:

        return None

def boundaryBox(img,contours,sheft):

    x, y, w, h = cv2.boundingRect(contours)


    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)


    # split sign from image

    sign = img[y+sheft:(y+h-sheft) , x+sheft:(x+w-sheft)]


    # return img, sign.copy(), ( ( x + w ) // 2 , ( y + h ) // 2 )

    return img, sign.copy(), ( x + ( w // 2 ) ,y + ( h // 2 ) )


model = load_model("model-3x3.h5")

labels = label_text('label_names.csv')


offset = 5

cap = cv2.VideoCapture(0)

ret , imgOrignal = cap.read()

scaling = 1

imgOrignal = cv2.resize(imgOrignal ,( imgOrignal.shape[1]//scaling , imgOrignal.shape[0]//scaling ) )

height = imgOrignal.shape[0]

width = imgOrignal.shape[1]


createRedTrackbar()
# createBlueTrackbar()

while(cap.isOpened()):

    ret , imgOrignal = cap.read()

    if ret ==False:
        break

    imgOrignal = cv2.resize(imgOrignal ,( imgOrignal.shape[1]//scaling , imgOrignal.shape[0]//scaling ) )

    imgOrignal = cv2.cvtColor(imgOrignal , cv2.COLOR_BGR2RGB)    

    vertices = np.array([ [int(width * 0.5), 0],[width ,0],[width,height],[int(width * 0.5),height] ])

    interestedImage = region_of_interest(imgOrignal,vertices=vertices)

    # lower_Red , upper_Red = getRed_Range()
    # lower_Blue , upper_Blue = getBlue_Range()

    # lower_red = np.array([180,50,20])

    # upper_red = np.array([180,255,255])

    redSelcted = selecRed_And_Blue_Pixels(interestedImage , np.array([150,50,20]) , np.array([170,255,255]) , np.array([110, 50 ,50]) , np.array([130,255,255]))

    blur , RGB , GRAY , thresh , dilated , canny = processImage(redSelcted)

    cv2.imshow('dilated', dilated)

    contours = findContour(canny)

    big = findBiggestContour(contours)

    # لان الصورة الأصلية رح ينرسم عليها خط

    # و انا بدي لما يقص صورة الساين تكون من دون خط

    img_With_Out_Line = interestedImage.copy()

    interestedImage_height = interestedImage.shape[0]

    interestedImage_width = interestedImage.shape[1]

    lineHeight = int ( interestedImage_height * 0.4 )

    cv2.line(interestedImage , ( 0 ,lineHeight ) , ( interestedImage_width , lineHeight ) , (0,0,250) ,4)
    
    if big is not None:

        contourArea = cv2.contourArea(big)        

        if(contourArea > 1500):

            print(contourArea)

            interestedImage , sign , (x,y) = boundaryBox(img_With_Out_Line , big ,3)

            cv2.line(interestedImage , ( 0 , lineHeight ) , (interestedImage_width , lineHeight ) , (255,0,0) , 4)

            if  ( y < ( lineHeight + offset ) ) or ( y > ( lineHeight - offset) )  :
            
                try :
                    sign = cv2.resize(sign, (32,32))

                    to_predict = sign.reshape(1,32,32,3)

                    scores = model.predict(to_predict)

                    cv2.imshow('sign' , sign)

                    # Scores is given for image with 43 numbers of predictions for each class

                    # Getting only one class with maximum value

                    prediction = np.argmax(scores)

                    # Getting labels

                    # Printing label for classified Traffic Sign

                    print('Label:', labels[prediction])

                    font = cv2.FONT_HERSHEY_DUPLEX

                    interestedImage = cv2.putText( interestedImage , labels[prediction]+"" , (0,20) , font ,0.3, (255,255,0) ,1 )

                except:
                    pass
    interestedImage = cv2.cvtColor(interestedImage,cv2.COLOR_RGB2BGR)

    cv2.imshow('Result' , interestedImage)

    if cv2.waitKey(2)  == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()






