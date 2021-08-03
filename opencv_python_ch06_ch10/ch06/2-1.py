import cv2
import numpy as np
import sys

image = cv2.imread('solidWhiteCurve.jpg')
mask = np.copy(image)

if image is None:
    print("There is no Image !")
    sys.exit

# BGR 제한 값 설정
# threshold (쓰레쉬홀드)값 설정
blue_threshold = 200
green_threshold = 200
red_threshold = 200
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# BGR 제한 값보다 작으면 검은색으로
# 이미지 (세로폭, 가로폭, 채널 수)
print(image.shape)
temp=(image[:,:,0]<bgr_threshold[0])
print(temp.shape)
# 검은색 영역은 True, 흰색 영역은 False가 나옴.
print(temp)
np.savetxt("text.csv",temp, fmt='%d',delimiter=',')
# image [:,:,0] => 모든 이미지 세로 , 가로 , 채널 0 (B채널) , 빛을 내는 모든 이미지를 의미
# BGR 채널 모두 Threshold 값보다 작은 픽셀 값은 True,
# 아닌 경우는 False로 배열 값을 저장
thresholds = (image[:,:,0] < bgr_threshold[0]) \
            | (image[:,:,1] < bgr_threshold[1]) \
            | (image[:,:,2] < bgr_threshold[2])
np.savetxt("thresholds.csv",thresholds, fmt='%d',delimiter=',')
# 마스크 배열 중 쓰레쉬홀드 값이 True 인 부분은 픽셀 값을 BGR 값울 0,0,0으로 설정
mask[thresholds] = [0,0,0]

cv2.imshow('white',mask) # 흰색 추출 이미지 출력
cv2.imshow('result',image) # 이미지 출력
cv2.waitKey(0)