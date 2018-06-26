import cv2
from EdgeDetector import auto_canny

def computeAspectRatio(w,h):
    if(w>h) :
        return w/h
    else:
        return h/w

def resize(image, width=None, height=None):
    if (height != None):
        ratio = height / image.shape[1]
    elif (width != None):
        ratio = width / image.shape[0]

    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
    return image

file="/Users/justin_ji/Desktop/OpenCV Test Image/cc1.jpg"
image = cv2.imread(file)
image = resize(image,800)
copied_image= image.copy();
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#blurred = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

edged = cv2.Canny(blurred, 30, 150)
#edged = auto_canny(blurred)

(_, cnts, _) = cv2.findContours(edged, 0,
       cv2.CHAIN_APPROX_SIMPLE)

#draw_cnts=[];

for cnt in cnts:
    (x,y,w,h) = cv2.boundingRect(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    #roi=sketch[y:y+h,x:x+w]
    if(w>40 and h>40 and h<80 and w<80 ):
        cv2.rectangle(copied_image,(x,y),(x+w,y+h),(0,255,0),2)
    elif(w>100 and h>100):
        print(w, " ", h, " ", computeAspectRatio(w, h))
        cv2.rectangle(copied_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

#cv2.drawContours(copied_image, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Image",copied_image)
cv2.imwrite("resources/cc2_op.jpg",copied_image)

cv2.waitKey(0)