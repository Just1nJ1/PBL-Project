import cv2
import numpy as np


img = cv2.imread("/Users/justin_ji/Desktop/OpenCV Test Image/WechatIMG1441.jpeg")

copy = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("gray", gray)

_, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

# cv2.imshow("Threshold", threshold)

_, cnts, _ = cv2.findContours(threshold, 0, cv2.CHAIN_APPROX_SIMPLE)

print(len(cnts))

rect = (0, 0, 0, 0)
wMax = 0
hMax = 0

for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if w * h > wMax * hMax:
        wMax = w
        hMax = h
        rect = (x, y, w, h)

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)


mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

img = img * mask2[:, :, np.newaxis]

cv2.imshow("12333", img)


kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

box = None

for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)

    if (w > 40 and h > 40 and w < 80 and h < 80):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif (w > 100 and h > 100 and w == wMax and h == hMax):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rect1 = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect1)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)


SrcPoints = np.float32(box[0:3])
CanvasPoints = np.float32([[0, img.shape[0]], [0, 0], [img.shape[1], 0]])

cv2.imshow("Image", img)

AffineMatrix = cv2.getAffineTransform(np.array(SrcPoints), np.array(CanvasPoints))
AffineImage = cv2.warpAffine(img, AffineMatrix, (img.shape[1], img.shape[0]))

cv2.imshow("AffineImage", AffineImage)


cv2.waitKey(0)
cv2.destroyAllWindows()