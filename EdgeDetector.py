import numpy as np
import cv2

def auto_blur(gray, rad = 11, resize = 1):
    blurred = cv2.GaussianBlur(gray, (rad, rad), 0)
    cv2.imshow("Blurred", cv2.resize(blurred, (blurred.shape[1]//resize, blurred.shape[0]//resize)))
    return blurred

def auto_median_blur(gray, rad = 11, resize = 1):
    blurred = cv2.medianBlur(gray, rad)
    cv2.imshow("Blurred", cv2.resize(blurred, (blurred.shape[1] // resize, blurred.shape[0] // resize)))
    return blurred

def auto_canny(blurred, sigma = 0.33, resize = 1, window_name = "Canny"):
    # compute the median of the single channel pixel intensities
    v = np.median(blurred)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)

    cv2.imshow(window_name, cv2.resize(edged, (edged.shape[1]//resize, edged.shape[0]//resize)))
    # return the edged image
    return edged

def auto_lap(blurred, resize = 1, window_name = "Laplacian"):
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv2.imshow(window_name, cv2.resize(lap, (lap.shape[1]//resize, lap.shape[0]//resize)))
    return lap

def auto_sobel(blurred, resize = 1, window_name = "Sobel"):
    sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    cv2.imshow(window_name, cv2.resize(sobelCombined, (sobelCombined.shape[1]//resize, sobelCombined.shape[0]//resize)))
    return sobelCombined