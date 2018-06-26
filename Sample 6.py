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

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

_, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

result = np.zeros(image.shape, np.uint8)

t = 0

while t < len(contours):
    area = cv2.contourArea(contours[t])
    rect = cv2.minAreaRect(contours[t])
    w = rect.shape[1]
    h = rect.shape[0]
    rate = min(w, h) / max(w, h)
    if rate > 0.85 and w < image.shape[1] / 4 and h < image.shape[0] / 4:
        continue
    t += 1

cv2.imshow("1", resize(threshold, 800))
cv2.waitKey(0)
cv2.destroyAllWindows()

# for (size_t t = 0; t < contours.size(); t++) {
# double area = contourArea(contours[t]);
# if (area < 100)
# continue;
# RotatedRect
# rect = minAreaRect(contours[t]);
# // 根据矩形特征进行几何分析
# float
# w = rect.size.width;
# float
# h = rect.size.height;
# float
# rate = min(w, h) / max(w, h);
# if (rate > 0.85 & & w < src.cols / 4 & & h < src.rows / 4) {
# printf("angle : %.2f\n", rect.angle);
# Mat qr_roi = transformCorner(src, rect);
# if (isXCorner(qr_roi) & & isYCorner(qr_roi)) {
# drawContours(src, contours, static_cast < int > (t), Scalar(0, 0, 255), 2, 8);
# imwrite(format("D:/gloomyfish/outimage/contour_%d.jpg", static_cast < int > (t)), qr_roi);
# drawContours(result, contours, static_cast < int > (t), Scalar(255, 0, 0), 2, 8);
# }
# }
# }
# imshow("result", src);
# imwrite("D:/gloomyfish/outimage/qrcode_patters.jpg", src);
# waitKey(0);
# return 0;