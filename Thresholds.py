import cv2
import numpy as np

def auto_Adaptive_Threshold_Gaussian(gray, size = 155, C = 1):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, C)

def auto_Adaptive_Threshold_Mean(gray, size = 155, C = 1):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, C)
