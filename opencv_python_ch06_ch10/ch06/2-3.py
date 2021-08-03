# -*- coding: utf-8 # UTF-8 encoding으로 변환

import cv2
import numpy as np
import sys

image = cv2.imread('slope_test.jpg')
# 이미지를 그레이스케일로 읽는다.
#image = cv2.imread('slope_test.jpg',cv2.IMREAD_GRAYSCALE)

if image is None:
    print("There is no Image!")
    sys.exit(-1)

height,width=image.shape[:2]

# 1. 그레이 이미지로 변경
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 2. 가우시안 블러(필터)
# 커널 사이즈를 키우면 블러가 높아짐
kernel_size=3
blur = cv2.GaussianBlur(gray, (kernel_size,kernel_size),0)
# 3. 캐니 영상처리
LOW_th, HIGH_th = 70,210
canny = cv2.Canny(blur,LOW_th,HIGH_th)
cv2.imshow('gray',gray)
cv2.imshow('blur',blur)
cv2.imshow('canny',canny)
cv2.waitKey(0)

