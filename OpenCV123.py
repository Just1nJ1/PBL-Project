import cv2
import numpy as np
import imutils
import mahotas
import datetime
from EdgeDetector import *
from FindingContours import *
import matplotlib.pyplot as plt
from Thresholds import *

def computeAspectRatio(w,h):
    if(w>h) :
        return w/h
    else:
        return h/w

def resize(image, width = None, height = None, scale = 1):
    if (width == None and height == None):
        return cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale))
    elif (height != None):
        ratio = height / image.shape[1]
    elif (width != None):
        ratio = width / image.shape[0]
    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
    return image

# frame = None
# cap = cv2.VideoCapture(0)
# while True:
#     _, frame = cap.read()
#
#     frame = resize(frame, 800)
#
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, threshold_frame = cv2.threshold(gray_frame, 170, 255, cv2.THRESH_BINARY)
#
#     (_, cnts, _) = cv2.findContours(threshold_frame, 0, cv2.CHAIN_APPROX_SIMPLE)
#
#     mask = np.zeros(frame.shape[:2], np.uint8)
#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)
#     rect = (0, 0, 0, 0)
#     wMax = 0
#     hMax = 0
#     cnt1 = None
#
#     for cnt in cnts:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#         if w * h > wMax * hMax:
#             wMax = w
#             hMax = h
#             rect = (x, y, w, h)
#             cnt1 = cnt
#
#     rect1 = cv2.minAreaRect(cnt1)
#     # calculate coordinate of the minimum area rectangle
#     box = cv2.boxPoints(rect1)
#     # normalize coordinates to integers
#     box =np.int0(box)
#     # 注：OpenCV没有函数能直接从轮廓信息中计算出最小矩形顶点的坐标。所以需要计算出最小矩形区域，
#     # 然后计算这个矩形的顶点。由于计算出来的顶点坐标是浮点型，但是所得像素的坐标值是整数（不能获取像素的一部分），
#     # 所以需要做一个转换
#     # draw contours
#     cv2.drawContours(frame, [box], 0, (0, 0, 255), 3)  # 画出该矩形
#
#
#
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     cv2.imshow('threshold', threshold_frame)
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) == 27:
#         break;
#

img = resize(cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/WechatIMG1442.jpeg"), 800)
copy = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
retval, threshold = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
# cv2.imshow("threshold", threshold)

(_, cnts, _) = cv2.findContours(threshold, 0, cv2.CHAIN_APPROX_SIMPLE)
# print(len(cnts))


mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (0, 0, 0, 0)
wMax = 0
hMax = 0
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if w * h > wMax * hMax:
        wMax = w
        hMax = h
        rect = (x, y, w, h)


cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
img = img*mask2[:,:,np.newaxis]
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# cv2.imshow("1233", img)
# cv2.imwrite("TempPics/1233.jpeg", img)


box = None
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    # roi=sketch[y:y+h,x:x+w]
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if (w > 40 and h > 40 and h < 80 and w < 80):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif (w > 100 and h > 100 and w == wMax and h == hMax):
        # print(w, " ", h, " ", computeAspectRatio(w, h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rect1 = cv2.minAreaRect(cnt)
        # calculate coordinate of the minimum area rectangle
        box = cv2.boxPoints(rect1)
        # normalize coordinates to integers
        box =np.int0(box)
        # 注：OpenCV没有函数能直接从轮廓信息中计算出最小矩形顶点的坐标。所以需要计算出最小矩形区域，
        # 然后计算这个矩形的顶点。由于计算出来的顶点坐标是浮点型，但是所得像素的坐标值是整数（不能获取像素的一部分），
        # 所以需要做一个转换
        # draw contours
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)  # 画出该矩形
        # print(box)



#SrcPoints = np.float32([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], [rect[0], rect[1] + rect[3]]])
SrcPoints = np.float32(box[0:3])
#CanvasPoints = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]]])
CanvasPoints = np.float32([[0, img.shape[0]], [0, 0], [img.shape[1], 0]])

# cv2.imshow("img", img)
#
# print(SrcPoints)
# print(CanvasPoints)

AffineMatrix = cv2.getAffineTransform(np.array(SrcPoints), np.array(CanvasPoints))
AffineImg = cv2.warpAffine(img, AffineMatrix, (img.shape[1], img.shape[0]))

# cv2.imshow("cnt", copy)
cv2.imshow("AffineImg", AffineImg)

gray_AffineImg = cv2.cvtColor(AffineImg, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray_AffineImg, 217, 255, cv2.THRESH_BINARY_INV)

final = cv2.bitwise_and(AffineImg, AffineImg, mask = binary)

# cv2.imshow("gray", gray_AffineImg)
#
# cv2.imshow("final", final)











cv2.waitKey(0)


# img = resize(cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/Coins and Cards.jpeg"), 800)
#
# # cv2.imshow("blur", blur)
# # canny = cv2.Canny(blur, 15, 90)
# # cv2.imshow("canny", canny)
#
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (328, 86, 1235-328, 726-86)
#
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
# img = img*mask2[:,:,np.newaxis]
#
# cv2.imshow("GrabCut", img)















# img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/Coins and Cards.jpeg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (3,3), )


# img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/Coins and Cards.jpeg")
# img = cv2.pyrMeanShiftFiltering(img, 25, 10)
# cv2.imshow("img", img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = auto_blur(img, 15, 2)
# img = auto_canny(img, resize = 2)
# auto_findContours(img, 2)


#
#
# image = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/cc1.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = auto_blur(gray, 5, 2)
# # print(image.shape)
# canny = cv2.Canny(blurred, 30, 150)
# (_, cnts, _) = auto_findContours(canny)

# auto_findContours(ATG, 2, "ATG")
# auto_findContours(ATM, 2, "ATM")
# auto_findContours(canny, 2, "Canny")
# auto_findContours(canny2, 2, "Canny2")
# auto_findContours(sobel, 2, "Sobel")
# auto_findContours(sobel2, 2, "Sobel2")
# auto_findContours(lap, 2, "Lap")
# auto_findContours(lap2, 2, "Lap2")



"""
cv2.drawContours(canny, cnts, -1, (255, 0, 0), 25)
cv2.imshow("Coins", resize(ATG, 2))
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print("Coin #{}".format(i + 1))
    coin = canny[y:y + h, x:x + w]
    cv2.imshow("Coin", coin)
    mask = np.zeros(canny.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))
    cv2.waitKey(0)
"""
#
# cap = cv2.VideoCapture(0)
# time = datetime.datetime.now()
# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     #cv2.imshow("Origin", cv2.resize(image,(image.shape[1]//2,image.shape[0]//2)))
#     cv2.imshow("Origin", frame)
#     canny = auto_canny(blurred)
#     #cv2.imshow("Canny", cv2.resize(image,(image.shape[1]//2,image.shape[0]//2)))
#     cv2.imshow("Canny", canny)
#     (_, cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     now = datetime.datetime.now()
#     if ( (now - time).seconds == 1):
#         time = now
#         print("Find {} coins".format(len(cnts)))
#         coins = frame.copy()
#         cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
#         cv2.imshow("Coins", coins)
#     if cv2.waitKey(1) == 27:
#         break;







"""



image = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/ThreeCoins.png")
#image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imshow("Blur", cv2.resize(blurred,(blurred.shape[1]//2,blurred.shape[0]//2)))

canny = cv2.Canny(blurred, 30, 150)
cv2.imshow("Canny", cv2.resize(canny, (canny.shape[1]//2,canny.shape[0]//2)))

(_, cnts, _) = cv2.findContours(canny.copy(), 0, cv2.CHAIN_APPROX_SIMPLE)

print("Find {} coins".format(len(cnts)))

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Coins", cv2.resize(coins, (coins.shape[1]//2, coins.shape[0]//2)))

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print("Coin #{}".format(i + 1))
    coin = image[y:y + h, x:x + w]
    cv2.imshow("Coin", coin)

    mask = np.zeros(image.shape[:2], dtype = "uint8")

    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))

    cv2.waitKey(0)



"""


"""
canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny1", cv2.resize(canny, (canny.shape[1]//5,canny.shape[0]//5)))
canny = cv2.Canny(image, 30, 200)
cv2.imshow("Canny2", cv2.resize(canny, (canny.shape[1]//5,canny.shape[0]//5)))
canny = cv2.Canny(image, 30, 140)
cv2.imshow("Canny3", cv2.resize(canny, (canny.shape[1]//5,canny.shape[0]//5)))
"""




"""

lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", cv2.resize(lap,(lap.shape[1]//5,lap.shape[0]//5)))

sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

sobleX = np.uint8(np.absolute(sobelX))
sobleY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("Sobel X", cv2.resize(sobelX,(sobleX.shape[1]//5,sobleX.shape[0]//5)))
cv2.imshow("Sobel Y", cv2.resize(sobelY,(sobleY.shape[1]//5,sobleY.shape[0]//5)))
cv2.imshow("Sobel Combined", cv2.resize(sobelCombined,(sobelCombined.shape[1]//5,sobelCombined.shape[0]//5)))








blurred = cv2.GaussianBlur(image, (5,5), 0)
cv2.imshow("Image", image)

T = mahotas.thresholding.otsu(blurred)
print("Otsu’s threshold: {}".format(T))


thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)
T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard", thresh)
"""



# cap = cv2.VideoCapture(0)
#
# while True:
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     lower_red = np.array([10,120,120])
#     upper_red = np.array([20,150,200])
#
#     # dark_yellow = np.uint8([[[164, 126, 76]]])
#     # dark_yellow = cv2.cvtColor(dark_yellow, cv2.COLOR_RGB2HSV)
#     # print(dark_yellow)
#
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     res = cv2.bitwise_and(frame, frame, mask = mask)
#
#     cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#     cv2.imshow("result", res)
#     if cv2.waitKey(1) == 27:
#         break;





"""


img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/bookpage.jpg")
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

cv2.imshow("original", img)
cv2.imshow("threshold", threshold)
cv2.imshow("threshold2", threshold2)
cv2.imshow("gaus", gaus)


img1 = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/pic1.jpg")
pic2 = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/pic2.jpg")
img2 = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/logo.jpg")

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(img)

img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
img1_fg = cv2.bitwise_and(img2, img2, mask = img)
final = cv2.add(img1_fg, img1_bg)
img1[0:rows, 0:cols] = final
cv2.imshow("img1", img1)
cv2.imshow("img", img)
cv2.imshow("mask_inv", mask_inv)
cv2.imshow("img1", img1)
cv2.imshow("Img1_bg",img1_bg)
cv2.imshow("Img1_fg",img1_fg)
cv2.imshow("final",final)

#cv2.imshow("Image",cv2.resize(img,(144*7,90*7)))

img = pic1 + pic2
img = cv2.add(pic1, pic2)
img = cv2.addWeighted(pic1, 0.8, pic2, 0.4, 0)

img = pic1[0:100,0:100]
print(type(img))

#image = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/123.jpg")

#print(image)

a=[[1,0,100],[0,1,50]]
print(type(a))
b=np.float32(a)
print(type(b))

img = np.zeros((300,300,3),dtype="uint8")

img[49:250,49:250,0:3] = 255

img = imutils.translate(img, 20, 20)

#img = imutils.rotate(img, 180, (img.shape[1] // 2, img.shape[0]//2))
#cv2.rectangle(img, (50,50), (250,250), (255,0,0), 0)
img = cv2.flip(img, 0)

cv2.line(img,(0,0),(200,200),(255,255,0),5)
cv2.line(img,(25,25),(225,225),(0,0,255),5)
cv2.line(img,(50,50),(250,250),(255,0,255),5)
cv2.line(img,(75,75),(275,275),(0,255,0),5)
cv2.line(img,(100,100),(300,300),(0,255,255),5)
cv2.line(img,(125,125),(300,300),(255,0,0),5)



for r in range (0,250,50):
    cv2.circle(img,(149,149),r,(r+20,r+10,r))

cv2.imshow("test",img)

cv2.waitKey(0)

cv2.destroyAllWindows()

image[0:100, 0:100] = (0, 255, 0)

print(image[0:100][0:100])


cv2.line(image, (0,0), (150,150), (255,255,255), 5)
print(image.dtype)
image[0:100, 0:100] = (0, 255, 0)
cv2.imshow("Updated", image)
"""

#cv2.imshow("Img", img)
cv2.destroyAllWindows()
