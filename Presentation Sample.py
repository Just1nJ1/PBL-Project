import cv2
import numpy as np

def resize(image, width=None, height=None):
    if (height != None):
        ratio = height / image.shape[1]
    elif (width != None):
        ratio = width / image.shape[0]

    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
    return image

img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/pic2.jpg")

img = resize(img, 800)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
_, threshold_inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

GaussianBlurred = cv2.GaussianBlur(img, (11, 11), 0)

MedianBlurred = cv2.medianBlur(img, 11)

Canny = cv2.Canny(gray, 100, 200)

Lap = cv2.Laplacian(gray, cv2.CV_64F)
Lap = np.uint8(np.absolute(Lap))

SobelX = cv2.Sobel(threshold, cv2.CV_64F, 1, 0)
SobelY = cv2.Sobel(threshold, cv2.CV_64F, 0, 1)
Sobel = cv2.bitwise_or(SobelX, SobelY)



cv2.imshow("Image", img)
cv2.imshow("Gray", gray)
cv2.imshow("Threshold", threshold)
cv2.imshow("Threshold_Inverse", threshold_inv)
cv2.imshow("GaussianBlur", GaussianBlurred)
cv2.imshow("MedianBlurred", MedianBlurred)
cv2.imshow("Canny", Canny)
cv2.imshow("Laplacian", Lap)
cv2.imshow("Sobel", Sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()