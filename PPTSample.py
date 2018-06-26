import cv2
import numpy as np

def resize(image, width = None, height = None, scale = 1):
    if (width == None and height == None):
        return cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale))
    elif (height != None):
        ratio = height / image.shape[1]
    elif (width != None):
        ratio = width / image.shape[0]
    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
    return image

image = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/WechatIMG1754.jpeg")
