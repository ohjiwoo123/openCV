import glob
import os
import cv2
import sys

interval = 1500 # 1.5 sec

base_path = '/Users/jwoh/opencvEx/'
img_path = os.path.join(base_path,'opencv_python_ch01_ch05/ch01/images/*.jpg')
img_files = glob.glob(img_path)
print(img_files)
print(len(img_files))

if not img_files:
    print("There is no img files")
    sys.exit()

# 이미지를 출력할 창을 생성
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# 이미지를 출력할 창의 크기를 화면 최대로 설정
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

idx = 0

while True:
    img = cv2.imread(img_files[idx])

    # 이미지를 읽지 못했다면
    if img is None:
        print('Image load failed')
        break

    cv2.imshow('image',img)

    # ESC 입력이 들어올 때까지 슬라이드쇼 진행
    if cv2.waitKey(interval)==27:
        break

    idx += 1
    if idx >= len(img_files):
        idx = 0

    cv2.destroyAllWindows()