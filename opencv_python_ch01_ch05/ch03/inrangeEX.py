import sys
import numpy as np
import cv2


src = cv2.imread('trafficLight.jpg')
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
#src = cv2.imread('candies2.png')

if src is None:
    print('Image load failed!')
    sys.exit()



src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# inRange = (입력이미지(src), lowerB,upperB)
# 여기서 src는 입력 이미지입니다. 'lowerb' 및 'upperb'는 임계값 영역의 하한 및 상한을 나타냅니다.
# 픽셀이 지정된 경계 내에 있으면 255로 설정되며 그렇지 않으면 0으로 설정됩니다. 이런 식으로 임계값 이미지를 반환합니다.

# 순서는 BGR 순서임
# 빨강등
#dst1 = cv2.inRange(src, (0, 0, 128), (128, 128, 255))
#dst2 = cv2.inRange(src_hsv, (0, 150, 0), (0, 255, 255))

# 주황등
#dst1 = cv2.inRange(src, (0, 128, 128), (128, 255, 255))
#dst2 = cv2.inRange(src_hsv, (11, 150, 0), (26, 255, 255))
# 초록등
#dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100))
#dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255))

#순서는 RGB 순서임
#빨강등
dst1 = cv2.inRange(src_rgb, (128, 0, 0), (255, 128, 128))
dst2 = cv2.inRange(src_hsv, (0, 150, 0), (0, 255, 255))

# #주황등
# dst1 = cv2.inRange(src_rgb, (0, 128, 128), (128, 255, 255))
# dst2 = cv2.inRange(src_hsv, (11, 150, 0), (26, 255, 255))
# #초록등
# dst1 = cv2.inRange(src_rgb, (0, 128, 0), (100, 255, 100))
# dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255))



cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()

cv2.destroyAllWindows()
