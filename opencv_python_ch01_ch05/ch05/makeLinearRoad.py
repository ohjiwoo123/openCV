import cv2
import numpy as np

import sys
import numpy as np
import cv2

pt1 = [0,0]
pt2 = [0,0]
pt3 = [0,0]
pt4 = [0,0]

counter = 0


def on_mouse(event, x,y, flags, param):
    global counter, pt1,pt2,pt3,pt4
    if flags & cv2.EVENT_FLAG_LBUTTON:
        if counter==0 and 1:
            pt1 = [x, y]
            print("pt1, x:{}, y:{}".format(x, y))
        elif counter==2 and 3:
            pt2 = [x, y]
            print("pt2, x:{}, y:{}".format(x, y))
        elif counter==4 and 5:
            pt3 = [x,y]
            print("pt3, x:{}, y:{}".format(x, y))
        elif counter==6 and 7:
            pt4 = [x,y]
            print("pt4, x:{}, y:{}".format(x, y))
        counter += 1

    elif flags & cv2.EVENT_FLAG_RBUTTON:
        counter -= 1

    if event == cv2.EVENT_MOUSEMOVE:
        print("x:{}, y:{}".format(x,y))


# 1. 정지 이미지 -> 동영상
def cropImage():
    global cropped_image
    global img_width
    global img_height
    img = cv2.imread('road.jpg')

    img_width = img.shape[1]
    img_height = img.shape[0]
    print('img_width:{}'.format(img_width))
    crop_pos_y = (img_height // 3) *2
    print(crop_pos_y)

    crop_pt_x = img_height
    cv2.imshow("original", img)

    # Cropping an image
    # 앞에 값이 height 뒤에 값이 width [height범위, width범위]
    cropped_image = img[crop_pos_y:img_height-1, 200:img_width-100]
    print(cropped_image.shape)
    # Display cropped image
    cv2.imshow("cropped", cropped_image)

    # Save the cropped image
    cv2.imwrite("Cropped Image.jpg", cropped_image)

def checkMousePos():
    global src
    # 마우스의 이벤트가 감지되면 on_mouse메소드가 호출
    src = cropped_image
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse, src)
    cv2.imshow('src', src)

    # 마우스 좌표값이 모두 입력되면 아무키나 눌러서 아래를 진행한다.
    cv2.waitKey()


def perspective():
    w, h = 720, 400
    print("pt1:{}, pt2:{}, pt3:{}, pt4:{}".format(pt1,pt2,pt3,pt4))
    srcQuad = np.array([pt1, pt2, pt3, pt4], np.float32)
    dstQuad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)

    # pers는 변환행렬 (3x3 matrix)
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (w, h))

    while True:
        cv2.imshow('src', src)
        cv2.imshow('dst', dst)

        keyValue = cv2.waitKey()
        if keyValue == 27:
            break

    cv2.destroyAllWindows()

cropImage()
checkMousePos()
perspective()