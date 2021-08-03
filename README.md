# MiniProject(Self Driving) - detect the Lane
## 1. 미니프로젝트 소개 
- 미니프로젝트의 주제는 차선을 찾는 것입니다. 여기에 한 단계 업그레이드 하여,<br>
도로에 하얀색 문자 ex) 청담대교 ↑와 같은 문자가 있을 때에도, 그 문자에 방해받지 않도록<br>
만드는 것을 목표로 하였습니다. <br>
유튜브에 있는 블랙박스 영상을 참고하여 미니프로젝트를 진행하였습니다.<br>
버전은 openCV 4.1 사용하였습니다.
## 2. ch06/miniproject(jw).py - Code
```python
# miniconda 이용하여 가상환경 세팅
# conda create -n opencv python=3.7
# pip install opencv-python==4.1.0.25
# conda activate opencv
# 참고 URL : https://velog.io/@bangsy/Python-OpenCV-4
# 유튜브 원본 URL : https://www.youtube.com/watch?v=ipyzW38sHg0
import pafy
import numpy as np
import cv2
import random

# 함수 정의

# 이미지를 RGB -> GRAY로
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 캐니 함수, 이미지에 캐니 적용하기
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
# 가우시안 블러 적용 함수
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
# 영역 설정 함수
def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 라인을 그리는 함수
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# 원을 그리는 함수
def draw_circle(img, lines, color=[0, 0, 255]):
    for line in lines:
        cv2.circle(img, (line[0], line[1]), 2, color, -1)

# hough -> 직선 변환 알고리즘
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_arr, lines)
    return lines

# 가중치 적용, 알파 , 베타, 사람인?
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

# 포인트 모으기
def Collect_points(lines):

    # reshape [:4] to [:2]
    interp = lines.reshape(lines.shape[0] * 2, 2)
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3] - line[1]) > 5:
            tmp = np.abs(line[3] - line[1])
            a = line[0];
            b = line[1];
            c = line[2];
            d = line[3]
            slope = (line[2] - line[0]) / (line[3] - line[1])
            for m in range(0, tmp, 5):
                if slope > 0:
                    new_point = np.array([[int(a + m * slope), int(b + m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
                elif slope < 0:
                    new_point = np.array([[int(a - m * slope), int(b - m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
    return interp

# 랜덤한 샘플 얻기
def get_random_samples(lines):
    one = random.choice(lines)
    two = random.choice(lines)
    if (two[0] == one[0]):  # extract again if values are overlapped
        while two[0] == one[0]:
            two = random.choice(lines)
    one, two = one.reshape(1, 2), two.reshape(1, 2)
    three = np.concatenate((one, two), axis=1)
    three = three.squeeze()
    return three

# 모델 파라미터 동작하기
def compute_model_parameter(line):
    # y = mx+n
    m = (line[3] - line[1]) / (line[2] - line[0])
    n = line[1] - m * line[0]
    # ax+by+c = 0
    a, b, c = m, -1, n
    par = np.array([a, b, c])
    return par

# 라인과 포인트 거리재기
def compute_distance(par, point):
    # distance between line & point

    return np.abs(par[0] * point[:, 0] + par[1] * point[:, 1] + par[2]) / np.sqrt(par[0] ** 2 + par[1] ** 2)

# 모델 확인
def model_verification(par, lines):
    # calculate distance
    distance = compute_distance(par, lines)
    # total sum of distance between random line and sample points
    sum_dist = distance.sum(axis=0)
    # average
    avg_dist = sum_dist / len(lines)

    return avg_dist

# 추정선을 그리기
def draw_extrapolate_line(img, par, color=(0, 0, 255), thickness=2):
    x1, y1 = int(-par[1] / par[0] * img.shape[0] - par[2] / par[0]), int(img.shape[0])
    x2, y2 = int(-par[1] / par[0] * (img.shape[0] / 2 + 100) - par[2] / par[0]), int(img.shape[0] / 2 + 100)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

# 정확한 라인 얻기
def get_fitline(img, f_lines):
    rows, cols = img.shape[:2]
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
    result = [x1, y1, x2, y2]

    #print("result= {} ".format(result))
    return result


# 정확한 라인 그리
def draw_fitline(img, result_l, result_r, color=(255, 0, 255), thickness=10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]), int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]), int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img, 1, 0.5)
    #print("final= {} ".format(final))
    return final


# 외곽선을 지우기
def erase_outliers(par, lines):
    # distance between best line and sample points
    distance = compute_distance(par, lines)

    # filtered_dist = distance[distance<15]
    filtered_lines = lines[distance < 13, :]
    return filtered_lines

# 부드럽게 만들기
def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0, 0, 0, 0])

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line

# 랜덤샘플 컨센서스 --> 이상치 검출방법 피팅(맞추기)
def ransac_line_fitting(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0, 0, 0])
    if (len(lines) != 0):
        for i in range(30):
            sample = get_random_samples(lines)
            parameter = compute_model_parameter(sample)
            cost = model_verification(parameter, lines)
            if cost < min:  # update best_line
                min = cost
                best_line = parameter
            if min < 3: break
        # erase outliers based on best line
        filtered_lines = erase_outliers(best_line, lines)
        fit_result = get_fitline(img, filtered_lines)
    else:
        if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
            l_fit_result = fit_result
            return l_fit_result
        else:
            r_fit_result = fit_result
            return r_fit_result

    if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
        l_fit_result = fit_result
        return l_fit_result
    else:
        r_fit_result = fit_result
        return r_fit_result


# 차선을 찾는 함수 ROI = 이미지 내 관심 영역
def detect_lanes_img(img):
    height, width = img.shape[:2]
    #print(height,width)

    # # Set ROI
    # vertices = np.array(
    #     [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
    #     dtype=np.int32)
    vertices = np.array(
        [[(210, height), ((width / 2 - 15), (height / 2)), (width / 2 + 45, height / 2), (width - 70, height)]],
        dtype=np.int32)
    ROI_img = region_of_interest(img, vertices)

    # Convert to grayimage
    # g_img = grayscale(img)

    # Apply gaussian filter
    blur_img = gaussian_blur(ROI_img, 3)

    # Apply Canny edge transform
    canny_img = canny(blur_img, 70, 210)
    # to except contours of ROI image
    vertices2 = np.array(
        [[(210, height), (width / 2 - 15, height / 2), (width / 2+ 45, height / 2), (width - 70, height)]],
        dtype=np.int32)
    canny_img = region_of_interest(canny_img, vertices2)

    # Perform hough transform
    # Get first candidates for real lane lines
    line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)

    # draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)
    # Get slope degree to separate 2 group (+ slope , - slope)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # ignore horizontal slope lines
    #print("horizontal before len(line_arr): {}".format(len(line_arr)))

    # 수평 경사도 160 보다 작게 설정
    line_arr = line_arr[np.abs(slope_degree) < 155]
    slope_degree = slope_degree[np.abs(slope_degree) < 155]
    #print("horizontal after len(line_arr): {}".format(len(line_arr)))

    # ignore vertical slope lines
    # 수직 경사도 95 보다 크게 설정

    line_arr = line_arr[np.abs(slope_degree) > 120]
    slope_degree = slope_degree[np.abs(slope_degree) > 120]
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    #print(line_arr.shape,'  ',L_lines.shape,'  ',R_lines.shape)

    # interpolation & collecting points for RANSAC
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines)

    # draw_circle(img,L_interp,(255,255,0))
    # draw_circle(img,R_interp,(0,255,255))

    # erase outliers based on best line
    left_fit_line = ransac_line_fitting(img, L_interp)
    right_fit_line = ransac_line_fitting(img, R_interp)

    # smoothing by using previous frames
    L_lane.append(left_fit_line), R_lane.append(right_fit_line)
    # lane이 10이 넘으면 smoothing 처리를 하라.
    if len(L_lane) > 10:
        left_fit_line = smoothing(L_lane, 10)
    if len(R_lane) > 10:
        right_fit_line = smoothing(R_lane, 10)

    # 최종 final = 최종 선을 그려라.
    final = draw_fitline(img, left_fit_line, right_fit_line)
    #print("final = {}".format(final))
    return final


# 참고 할 유튜브 URL 입력
url = "https://www.youtube.com/watch?v=ipyzW38sHg0"
# url을 pafy하라
video = pafy.new(url)
# result랑 lane을 각각 배열로 선언
fit_result, l_fit_result, r_fit_result, L_lane, R_lane = [], [], [], [], []
# best = 비디오를 mp4타입으로
best = video.getbest(preftype="mp4")
# cap = best.url로 비디오캡쳐
cap = cv2.VideoCapture(best.url)
# 동영상 크기(frame정보)를 읽어옴
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(frameWidth,frameHeight)
# 동영상 프레임을 캡쳐
frameRate = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (frameWidth, frameHeight)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# 파일 경로 설정
out1Path = '/Users/jwoh/opencvEx/opencv_python_ch06_ch10/ch06/Road(jw).mp4'
# out1 파일저장
out = cv2.VideoWriter(out1Path, fourcc, frameRate, frame_size)

# q 누를 때 까지 반복 실행
while (cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    # if frame.shape[0] != 540:  # resizing for challenge video
    #     frame = cv2.resize(frame, None, fx=3 / 4, fy=3 / 4, interpolation=cv2.INTER_AREA)
    result = detect_lanes_img(frame)
    # print(frame)

    out.write(result)
    # 화면 보기
    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 3. 오류 해결 내역 
- ROI_IMG 영역 및 Canny_IMG 영역을 수정하여 특정 범위 내에서만 Lane을 찾도록 하였다
- 저장이 안 되어서 헤맨 결과 프레임이 다르지 않아서 생긴 문제였고, 이전에 배운 youtube.py를 참고하여 저장하기에 성공했다.<br>
처음에 out.write(frame)으로 저장해서 왜 drawing line이 안 보이지? 고민하다 out.write(result)로 바꾸니 잘 나왔다..!
- 이번 프로젝트의 핵심은 개인적으로  slope_degree라고 생각하는데 수평 각도와 수직 각도를 제한시킴으로써, 선이 다른 쪽으로 유인 안 되게끔 하였다.
## 4. 아쉬운 점 
- 아직 코드에 대한 이해도가 부족하고, 다른 좋은 기술들도 많을 테지만 지식이 부족하여 사용하지 못하였다.
- 좀 더 집중했더라면 좋았을텐데, 개인적인 사정으로 집중력이 좋지 않아서 아쉬웠다. 
- 아직 이 분야에 대한 전체적인 이해도와 용어 등의 기초 지식들을 쌓아 나갈 필요가 있겠다. 
- github pull하고 push 해야하는데 아직 익숙하지 못해서 계속 강제 push 중인데 뭔가 아쉽다.
## 5. 참고 사이트 정리
- https://velog.io/@bangsy/Python-OpenCV-4 (파이썬 유튜브 링크로 영상 처리하기)
- https://www.youtube.com/watch?v=ipyzW38sHg0 (도로 위 영상, 유튜브 원본)
- https://m.blog.naver.com/windowsub0406/220893893795 (SelfDriving 네이버블로그)
- https://github.com/windowsub0406/SelfDrivingCarND (SelfDriving Github)
