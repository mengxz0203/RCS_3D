import cv2


def take_photo():
    index = 0
    cap = cv2.VideoCapture(index + cv2.CAP_DSHOW)

    # 设置视频流宽高
    width = 2560
    height = 720
    mid = width // 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 获取视频流宽高
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    flag = 1

    while True:
        ret, frame = cap.read()
        cv2.imshow('img', frame)
        cv2.waitKey(10)
        left_img = frame[:, :mid]
        right_img = frame[:, mid:]
        cv2.imshow('left_img', left_img)
        cv2.imshow('right_img', right_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('../images/test1/left/left' + str(flag) + '.jpg', left_img)
            cv2.imwrite('../images/test1/right/right' + str(flag) + '.jpg', right_img)
            flag = flag + 1

    cap.release()
    cv2.destroyAllWindows()


'''    
    flag = 0
    while flag < 10:
        ret, frame = cap.read()
        cv2.imshow('img', frame)
        left_img = frame[:, :mid]
        right_img = frame[:, mid:]
        cv2.imshow('left_img', left_img)
        cv2.imwrite('../images/test/left/left.jpg', left_img)
        cv2.imshow('right_img', right_img)
        cv2.imwrite('../images/test/right/right.jpg', right_img)
        flag = flag + 1
'''

take_photo()
