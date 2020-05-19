import cv2

index = 0
cap = cv2.VideoCapture(index + cv2.CAP_DSHOW)

# 设置视频流宽高
width = 2560
height = 720
mid = width / 2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 获取视频流宽高
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


cv2.namedWindow("left_img")
cv2.namedWindow("right_img")
cv2.resizeWindow("left_img", 1280, 720)
cv2.resizeWindow("right_img", 1280, 720)
while True:
    ret, frame = cap.read()
    cv2.imshow("img", frame)
    cv2.waitKey(0)
    left_img = frame[:, :mid]
    right_img = frame[:, mid:]
    cv2.imshow("left_img", left_img)
    cv2.waitKey(0)
    cv2.imshow("right_img", right_img)
    cv2.waitKey(0)
