import cv2

# src = cv2.imread('trafficLight.jpg')
src = cv2.imread('trafficLight2.jpg')
src_rgb = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)

def on_trackbar(pos):
    rmin = cv2.getTrackbarPos('R_min','dst')
    rmax = cv2.getTrackbarPos('R_max','dst')
    gmin = cv2.getTrackbarPos('G_min','dst')
    gmax = cv2.getTrackbarPos('G_max','dst')
    bmin = cv2.getTrackbarPos('B_min','dst')
    bmax = cv2.getTrackbarPos('B_max','dst')

    dst = cv2.inRange(src_rgb, (rmin,gmin,bmin), (rmax,gmax,bmax))
    cv2.imshow('dst',dst)

cv2.imshow('src',src)
cv2.namedWindow('dst')


cv2.createTrackbar('R_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('R_max', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('G_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('G_max', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('B_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('B_max', 'dst', 0, 255, on_trackbar)

on_trackbar(0)
cv2.waitKey()

