# 2-2는 2-1에서 ROI설정 부분이 추가
import cv2  # opencv 사용
import numpy as np
import sys

# 영역 관리 설정
ROI_DISP=True
def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅
    # img라는 배열을 참조해서 img.shape -> height, width, channel을 읽어와서
    # 그와 동일한 배열 mask를 생성한다.
    # zeros_like -> height, width, channel 배열 크기로 0으로 채워진 배열을 생성
    # ones_like -> height, width, channel 배열 크기로 1로 채워진 배열을 생성
    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)


    # 이미지와 color로 채워진 ROI를 합침
    # bitwise 1이 흰색 0이 검정
    ROI_image = cv2.bitwise_and(img, mask)
    # cv2.show('test',mask) -> 안에 빨갛게 칠해짐.
    # cv2.waitKey()
    return ROI_image


def mark_img(img, blue_threshold=200, green_threshold=200, red_threshold=200):  # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (image[:, :, 0] < bgr_threshold[0]) \
                 | (image[:, :, 1] < bgr_threshold[1]) \
                 | (image[:, :, 2] < bgr_threshold[2])
    mark[thresholds] = [0, 0, 0]
    return mark


image = cv2.imread('solidWhiteCurve.jpg')  # 이미지 읽기


if image is None:
    print("There is no Image !")
    sys.exit

height, width = image.shape[:2]  # 이미지 높이, 너비

# 사다리꼴 모형의 Points
vertices = np.array(
    [[(50, height),
      # 왼쪽차선
      (width / 2 - 45, height / 2 + 60),
      # 오른쪽 차선
      (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
    dtype=np.int32)

# ROI를 위한 마스크를 확인하기 위한 디버깅 코드
cv2.polylines(image,vertices,True,(255,0,0),2)
cv2.imshow('image',image)
cv2.waitKey()

roi_img = region_of_interest(image, vertices, (0, 0, 255))  # vertices에 정한 점들 기준으로 ROI 이미지 생성

# mark = np.copy(roi_img)  # roi_img 복사
# mark = mark_img(roi_img)  # 흰색 차선 찾기

# 흰색 차선 검출한 부분을 원본 image에 overlap 하기
# color_thresholds = (mark[:, :, 0] == 0) & (mark[:, :, 1] == 0) & (mark[:, :, 2] > 200)
# image[color_thresholds] = [0, 0, 255]
#
# cv2.imshow('roi_white', mark)  # 흰색 차선 추출 결과 출력
# cv2.imshow('result', image)  # 이미지 출력
# cv2.waitKey(0)
