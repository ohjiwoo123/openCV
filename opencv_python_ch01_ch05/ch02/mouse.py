import sys
import numpy as np
import cv2


oldx = oldy = -1

def on_mouse(event, x, y, flags, param):
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
        oldx, oldy = x, y
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))

    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
        print('EVENT_LBUTTONUP: %d, %d' % (x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y
        elif flags & cv2.EVENT_FLAG_MBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y
WIN_WIDTH = 1024
WIN_HEIGHT = 768
CHANNEL = 3

# 비어있는 캔버스 생성
img = np.ones((WIN_HEIGHT, WIN_WIDTH, CHANNEL), dtype=np.uint8) * 255

cv2.namedWindow('image')
#마우스 이벤트를 처리하는 사용자함수 on_mouse를 Callback함수로 등록
cv2.setMouseCallback('image', on_mouse, img)

cv2.imshow('image', img)
cv2.waitKey()

cv2.destroyAllWindows()