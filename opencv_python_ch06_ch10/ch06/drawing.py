import numpy as np
import cv2

img = np.full((400, 400, 3), 255, np.uint8)

# 도형이나 직선을 그릴 때는 점의 좌표
# 점의 좌표 tuple로 표현한다.
cv2.line(img, (50, 50), (200, 50), (0, 0, 255), 5)  # 끝에 5가 두께
cv2.line(img, (50, 60), (150, 160), (0, 0, 128))

# 사각형을 그릴 때 2가지 방법
# 시작점의 좌표, width, height, 색상, 두께
cv2.rectangle(img, (50, 200, 150, 100), (0, 255, 0), 2)
# 시작점의 좌표, 종점의 좌표, 색상, 두께를 -1로 설정하면 사각형 안을 채워준다.
cv2.rectangle(img, (70, 220), (180, 280), (0, 128, 0), -1)

# 중심점의 좌표, 반지름, 색상, 두께, 라인스타일
cv2.circle(img, (300, 100), 30, (255, 255, 0), -1, cv2.LINE_AA)
cv2.circle(img, (300, 100), 60, (255, 0, 0), 3, cv2.LINE_AA)

# 다각형의 좌표 값을 주고
pts = np.array([[250, 200], [300, 200], [350, 300], [250, 300]])
cv2.polylines(img, [pts], True, (255, 0, 255), 2)

# 캔버스에 문자열 추가
text = 'Hello? OpenCV ' + cv2.__version__
# 문자열의 시작점 좌표, 폰트, 글자의 크기, 색상, 두께
cv2.putText(img, text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, # 0.8 = 글자크기
            (0, 0, 255), 1, cv2.LINE_AA)    # 1 = 글자 두께

cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

