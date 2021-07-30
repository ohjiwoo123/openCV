import sys
import numpy as np
import cv2

# src = 캔디사진
src = cv2.imread('candies.png')

# src가 없으면 이미지로드 실패 출력 후 시스템 종료
if src is None:
    print('Image load failed!')
    sys.exit()

# src_hsv는 src의 컬러를 BGR -> HSV
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# 트랙바 함수
def on_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min', 'dst')
    hmax = cv2.getTrackbarPos('H_max', 'dst')

    # cv2.inRange(src,lowerb,upperb,dst=None) -> dst
    # src : 입력 행렬, lowerb : 하한 값 행렬 또는 스칼라, upperb: 상한 값 행렬 또는 스칼라
    # dst : 입력 영상과 같은 크기의 마스크 영상(numpy.unit8)
    dst = cv2.inRange(src_hsv, (hmin, 150, 0), (hmax, 255, 255))
    cv2.imshow('dst', dst)


cv2.imshow('src', src)
cv2.namedWindow('dst')

cv2.createTrackbar('H_min', 'dst', 50, 179, on_trackbar)
cv2.createTrackbar('H_max', 'dst', 80, 179, on_trackbar)
on_trackbar(0)

cv2.waitKey()

cv2.destroyAllWindows()
