import cv2
import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 480

# 비어있는 캔버스의 배열을 생성
img = np.zeros((IMG_HEIGHT,IMG_WIDTH),np.uint8)
print(img.shape)

# 창을 생성한다. 창의 이름을 지정하여
# trackbar를 추가하기 위해,
# trackbar를 추가하고, 초기값도 설정
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',IMG_WIDTH,IMG_HEIGHT)

def on_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min','dst')
    hmax = cv2.getTrackbarPos('H_max','dst')
    smin = cv2.getTrackbarPos('S_min','dst')
    smax = cv2.getTrackbarPos('S_max','dst')
    vmin = cv2.getTrackbarPos('V_min','dst')
    vmax = cv2.getTrackbarPos('V_max','dst')

    dst = cv2.inRange(img, (hmin,smin,vmin), (hmax,smax,vmax))
    cv2.imshow('dst',dst)

cv2.imshow('img',img)
cv2.namedWindow('dst')

cv2.createTrackbar('H_min', 'dst', 0, 180, on_trackbar)
cv2.createTrackbar('H_max', 'dst', 0, 180, on_trackbar)
cv2.createTrackbar('S_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('S_max', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('V_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('V_max', 'dst', 0, 255, on_trackbar)

on_trackbar(0)
cv2.waitKey()