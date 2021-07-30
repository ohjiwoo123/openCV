import cv2

# src = cv2.imread('trafficLight.jpg')
src = cv2.imread('trafficLight.jpg')
src_hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)

def on_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min','dst')
    hmax = cv2.getTrackbarPos('H_max','dst')
    smin = cv2.getTrackbarPos('S_min','dst')
    smax = cv2.getTrackbarPos('S_max','dst')
    vmin = cv2.getTrackbarPos('V_min','dst')
    vmax = cv2.getTrackbarPos('V_max','dst')

    # inRange = (입력이미지(src), lowerB,upperB)
    # 여기서 src는 입력 이미지입니다. 'lowerb' 및 'upperb'는 임계값 영역의 하한 및 상한을 나타냅니다.
    # 픽셀이 지정된 경계 내에 있으면 255로 설정되며 그렇지 않으면 0으로 설정됩니다. 이런 식으로 임계값 이미지를 반환합니다.
    dst = cv2.inRange(src_hsv, (hmin,smin,vmin), (hmax,smax,vmax))
    cv2.imshow('dst',dst)

cv2.imshow('src',src)
cv2.namedWindow('dst')

cv2.createTrackbar('H_min', 'dst', 0, 180, on_trackbar)
cv2.createTrackbar('H_max', 'dst', 0, 180, on_trackbar)
cv2.createTrackbar('S_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('S_max', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('V_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('V_max', 'dst', 0, 255, on_trackbar)

on_trackbar(0)
cv2.waitKey()