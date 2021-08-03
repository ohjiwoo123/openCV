# -*- coding: utf-8 -*- #  UTF-8 encoding으로 변환

import cv2  # opencv
import numpy as np
import sys

def on_low_th(pos):
    global blur, HIGH_th, canny
    LOW_th = pos
    canny = cv2.Canny(blur, LOW_th, HIGH_th)
    cv2.imshow('canny', canny)

def on_high_th(pos):
    global blur, LOW_th, canny
    HIGH_th = pos
    canny = cv2.Canny(blur, LOW_th, HIGH_th)
    cv2.imshow('canny', canny)

image = cv2.imread('slope_test.jpg')
if image is None:
    print("Image File is not read!")
    sys.exit(-1)

height, width = image.shape[:2]
# 1. 그레이 이미지로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 가우시안 블러(필터)
# kernel_size를 키울수록 blur는 심해진다.
kernel_size = 3
blur = cv2.GaussianBlur(gray, (kernel_size,kernel_size),0)

# 3. trackbar를 추가하기 위해 canny window를 먼저 생성
cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('canny', width, height)

cv2.createTrackbar('low_th', 'canny', 0, 99, on_low_th)
cv2.createTrackbar('high_th', 'canny', 80, 255, on_high_th)

# 4. canny 영상처리


LOW_th, HIGH_th = 70, 210
canny = cv2.Canny(blur, LOW_th, HIGH_th)

cv2.imshow('gray', gray)
cv2.imshow('blur', blur)
cv2.imshow('canny',canny)
cv2.waitKey(0)

cv2.destroyAllWindows()