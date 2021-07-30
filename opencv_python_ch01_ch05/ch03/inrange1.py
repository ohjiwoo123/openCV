import sys
import numpy as np
import cv2


src = cv2.imread('candies.png')
#src = cv2.imread('candies2.png')

if src is None:
    print('Image load failed!')
    sys.exit()

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# inRange = (입력이미지(src), lowerB,upperB)
# 여기서 src는 입력 이미지입니다. 'lowerb' 및 'upperb'는 임계값 영역의 하한 및 상한을 나타냅니다.
# 픽셀이 지정된 경계 내에 있으면 255로 설정되며 그렇지 않으면 0으로 설정됩니다. 이런 식으로 임계값 이미지를 반환합니다.
dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100))
dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255))

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()

cv2.destroyAllWindows()
