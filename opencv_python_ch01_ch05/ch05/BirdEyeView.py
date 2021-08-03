import sys
import numpy as np
import cv2

# 정지 이미지와 동영상을 모두 처리 할수 있는 형태
# 입력되는 이미지가 동영상인지, 정지 이미지인지 설정하는 플래그
video = True

# perspective를 위한 4개 좌표값 초기화
pt1 = [0,0]
pt2 = [0,0]
pt3 = [0,0]
pt4 = [0,0]

# 마우스 클릭 카운터 초기화
counter = 0

# on_mouse 함수 인자에 4개가 원래 고정 값인지..?
def on_mouse(event, x,y, flags, param):
    # pt1,pt2,pt3,pt4 전역변수로 선언
    global counter, pt1,pt2,pt3,pt4
    # 만약에 flags 그리고 마우스왼쪽버튼클릭이 일어난다면
    if flags & cv2.EVENT_FLAG_LBUTTON:
        # 현재 클릭이 2번으로 인식되어 0 and 1로 하였음.
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

img_filename = 'road.jpg'
video_filename = 'project_video.mp4'
w, h = 200, 350

if video==True:
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print("Video open failed!")
        sys.exit()

    # 비디오 프레임 크기, 전체 프레임수, FPS 등 출력
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000 / video_fps)
    print('video_width:{}'.format(video_width))
    print('video_height:{}'.format(video_height))
    print('video_fps:{}'.format(video_fps))

    # 비디오 첫 프레임 읽어오기
    ret, src = cap.read()

    if not ret:
        sys.exit()

    # 정지 이미지로 저장
    # cv2.imwrite(img_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # 마우스의 이벤트가 감지되면 on_mouse메소드가 호출
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse, src)

    cv2.imshow('src', src)

    # 마우스 좌표값이 모두 입력되면 아무키나 눌러서 아래를 진행한다.
    cv2.waitKey()

    print("pt1:{}, pt2:{}, pt3:{}, pt4:{}".format(pt1,pt2,pt3,pt4))

    # pt1~pt4까지는 시계방향으로 마우스로 좌표 지정
    srcQuad = np.array([pt1, pt2, pt3, pt4], np.float32)
    dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)

    #pers는 변환행렬 (3x3 matrix)
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (w, h))

    while True:
        # 비디오 첫 프레임 읽어오기
        ret, src = cap.read()

        if not ret:
            break
        dst = cv2.warpPerspective(src, pers, (w, h))
        cv2.imshow('src', src)
        cv2.imshow('dst', dst)

        keyValue = cv2.waitKey(delay)
        if keyValue==27:
            break

    #
    cap.release()


else:
    # 정지 이미지 읽기
    src = cv2.imread(img_filename)

    if src is None:
        print('Image load failed!')
        sys.exit()

    # 마우스의 이벤트가 감지되면 on_mouse메소드가 호출
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse, src)

    cv2.imshow('src', src)

    # 마우스 좌표값이 모두 입력되면 아무키나 눌러서 아래를 진행한다.
    cv2.waitKey()

    print("pt1:{}, pt2:{}, pt3:{}, pt4:{}".format(pt1,pt2,pt3,pt4))

    # pt1~pt4까지는 시계방향으로 마우스로 좌표 지정
    srcQuad = np.array([pt1, pt2, pt3, pt4], np.float32)
    dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)

    #pers는 변환행렬 (3x3 matrix)
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (w, h))

    while True:
        cv2.imshow('src', src)
        cv2.imshow('dst', dst)

        keyValue = cv2.waitKey()
        if keyValue==27:
            break

cv2.destroyAllWindows()