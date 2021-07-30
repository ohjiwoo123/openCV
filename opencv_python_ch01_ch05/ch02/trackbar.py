import numpy as np
import cv2


def on_level_change(pos):
    value = pos * 16
    if value >= 255:
        value = 255

    # 캔버스의 색깔을 바꾼다.
    img[:] = value
    cv2.imshow('image', img)
    print("call me")
    print("pos={}".format(pos))

# 검은색 캔버스 480 x 640
img = np.zeros((480, 640), np.uint8)

# 'image' 창을 생성
cv2.namedWindow('image')
cv2.resizeWindow('image',640,480)

# 'image'창 안에 Trackbar 추가 (trackbar의 이름은 'level')
# trackbar가 움직이면 on_level_change 함수가 호출
cv2.createTrackbar('level', 'image', 0, 16, on_level_change)

cv2.imshow('image', img)
# 아무키나 들어왔을 때
cv2.waitKey()
cv2.destroyAllWindows()
