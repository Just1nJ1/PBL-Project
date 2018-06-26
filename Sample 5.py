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

# img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/WechatIMG1755.jpeg")
img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/WechatIMG287.jpeg")
blur=cv2.GaussianBlur(img,(0,0),3)
image=cv2.addWeighted(img,1.5,blur,-0.5,0)
cv2.imshow("image", image)
cv2.imshow("img", img)
cv2.waitKey(0)
# img = resize(img, 800)
# img = cv2.GaussianBlur(img, (3, 3), 0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # cv2.imshow("gray", gray)
#
# _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
#
# # cv2.imshow("Threshold", threshold)
#
# _, cnts, _ = cv2.findContours(threshold, 0, cv2.CHAIN_APPROX_SIMPLE)
#
# print(len(cnts))
#
# rect = (0, 0, 0, 0)
# wMax = 0
# hMax = 0
#
# for cnt in cnts:
#     (x, y, w, h) = cv2.boundingRect(cnt)
#     if w * h > wMax * hMax:
#         wMax = w
#         hMax = h
#         rect = (x, y, w, h)
#
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
#
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
#
#
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
#
# img = img * mask2[:, :, np.newaxis]
#
# cv2.imshow("12333", img)






gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold", threshold)
edges = cv2.Canny(threshold, 100, 200)
# edges = cv2.Laplacian(gray, cv2.CV_64F)

cv2.imshow("edges", edges)
# cv2.waitKey(0)
img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.waitKey(0)
hierarchy = hierarchy[0]
found = []
for i in range(len(contours)):
    k = i
    c = 0
    while hierarchy[k][2] != -1:
        k = hierarchy[k][2]
        c = c + 1
    if c >= 3:
        found.append(i)

for i in found:
    img_dc = img.copy()
    cv2.drawContours(img_dc, contours, i, (0, 255, 0), 3)
    # cv2.imshow("img", img_dc)
    # cv2.waitKey(0)

draw_img = img.copy()
boxes = np.array([None])

for i in found:
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if boxes.all() == None:
        boxes = box
    else:
        boxes = np.vstack((boxes, box))
    cv2.drawContours(draw_img,[box], 0, (0,0,255), 2)
# cv2.imshow('draw_img', draw_img)

rect = cv2.minAreaRect(boxes)
box = cv2.boxPoints(rect)
box = np.array(box)
draw_img = img.copy()
cv2.polylines(draw_img, np.int32([box]), True, (0, 0, 255), 2)
print(box)
# cv2.imshow('final', draw_img)
# cv2.imshow('img', img)

SrcPoints = np.float32(box[0:3])
CanvasPoints = np.float32([[img.shape[1], img.shape[0]], [0, img.shape[0]], [0, 0]])

AffineMatrix = cv2.getAffineTransform(np.array(SrcPoints), np.array(CanvasPoints))
AffineImg = cv2.warpAffine(img, AffineMatrix, (img.shape[1], img.shape[0]))

cv2.imshow("AffineImg", AffineImg)
########################################################################################################################
width = ((box[2][0] - box[3][0]) ** 2 + (box[2][1] - box[3][1]) ** 2) ** 0.5
height = ((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2) ** 0.5

ratio = width / height

finalH = img.shape[0]
finalW = finalH * ratio

SrcPoints_2 = np.float32(box[0:3])
CanvasPoints_2 = np.float32([[finalW, finalH], [0, finalH], [0, 0]])

AffineMatrix_2 = cv2.getAffineTransform(np.array(SrcPoints_2), np.array(CanvasPoints_2))
AffineImg_2 = cv2.warpAffine(img, AffineMatrix_2, (np.int32(finalW), np.int32(finalH)))

cv2.imshow("AffineImg_2", resize(AffineImg_2, 800))

# cv2.waitKey(0)
#
# i = 0
#
# contour_all = np.array([])
# while i < 3:
#     c = contours[i]
#     i += 1
#     for sublist in c:
#         for p in sublist:
#             # contour_all.append(p)
#             np.vstack((contour_all, p))
# rect = cv2.minAreaRect(contour_all)
# box = cv2.boxPoints(rect)
# box = np.array(box)
# draw_img = img.copy()
# cv2.polylines(draw_img, np.int32([box]), True, (0, 0, 255), 10)
# cv2.imshow("final", draw_img)


cv2.waitKey(0)
cv2.destroyAllWindows()