import sys
import cv2

# 파일경로
filePath = '/Users/jwoh/opencvEx/opencv_python_ch01_ch05/ch02/video2.mp4'
# 카메라 열기
cap = cv2.VideoCapture(filePath)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 카메라 프레임 크기 출력
print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 카메라 프레임 처리
while True:
    ret, frame = cap.read()
    resize_frame = cv2.resize(frame, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    #print(resize_frame.shape)
    if not ret:
        break

    inversed = ~resize_frame  # 반전

    win_xpos = 100
    win_ypos = 100
    cv2.imshow('frame', resize_frame)
    cv2.imshow('inversed', inversed)
    cv2.moveWindow('frame', win_xpos, win_ypos)
    cv2.moveWindow('inversed', win_xpos+(resize_frame.shape[1]), win_ypos)

    if cv2.waitKey(100) == 27:
        break

cap.release()
cv2.destroyAllWindows()
