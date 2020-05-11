import cv2
import numpy as np

thresholdBlocksize = 11


def show_img(img_name):
    cv2.imshow("window", img_name)
    cv2.waitKey()


def find_3d_screen_image_region(img):
    # cv2.imshow("img", img)

    # 缩小尺寸
    shrinkedPic = cv2.pyrDown(img)
    showPic = cv2.pyrDown(shrinkedPic)
    shrinkedPic = cv2.pyrDown(shrinkedPic)
    # cv2.imshow("img", shrinkedPic)
    # cv2.waitKey()

    # 灰度化
    gray = cv2.cvtColor(shrinkedPic, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey()

    # 中值滤波降噪
    median = cv2.medianBlur(gray, 9)
    show_img(median)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # show_img(blur)

    # 二值化
    # ret, binary = cv2.threshold(median, 40, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresholdBlocksize, 2)
    # ret, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print("ret", ret)
    show_img(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    erode = cv2.erode(binary, kernel)
    show_img(erode)

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 对轮廓图执行一次闭运算，加强轮廓的连通性
    dst = cv2.dilate(erode, kernel_1)
    show_img(dst)

    # 边缘检测
    canny = cv2.Canny(dst, 200, 2.5)
    show_img(canny)

    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_2)
    show_img(closed)

    find, contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 从轮廓图中提取出所有轮廓的数据
    # cv2.imshow("find", find)
    # cv2.waitKey(0)

    h, w = dst.shape[:2]
    print("h:", h)
    print("w:", w)
    linePic = np.zeros((h, w, 3))

    cv2.drawContours(linePic, contours, -1, (0, 0, 255), 1)
    show_img(linePic)

    cv2.drawContours(shrinkedPic, contours, -1, (0, 0, 255), 1)
    show_img(shrinkedPic)

    maxArea = 0
    length = len(contours)
    for index in range(length):
        if cv2.contourArea(contours[index]) > cv2.contourArea(contours[maxArea]):
            maxArea = index

    polyContours = cv2.approxPolyDP(contours[maxArea], 10, True)

    x = []
    hull_list = []
    length = len(polyContours)
    for index in range(length):
        hull = cv2.convexHull(polyContours[index])
        hull_list.append(hull[0][0])
        print(hull[0][0][0], hull[0][0][1])
        coords = tuple([hull[0][0][0], hull[0][0][1]])
        x.append(coords)
    print("hull_list:", hull_list)
    print(x)

    polyPic = np.zeros((h, w, 3))
    # cv2.drawContours(polyPic, polyContours, -1, (0, 255, 0), 1)
    cv2.polylines(polyPic, [polyContours], True, (0, 0, 255), 2)
    show_img(polyPic)

    cv2.polylines(shrinkedPic, [polyContours], True, (0, 255, 0), 2)
    show_img(shrinkedPic)

    points1 = np.float32([x[0], x[1],  x[2], x[3]])
    # points1 = np.float32([])
    points2 = np.float32([[0, 0], [0, h],  [w, h], [w, 0]])

    # 计算得到转换矩阵
    M = cv2.getPerspectiveTransform(points1, points2)

    print("M:", M)

    # 实现透视变换转换
    processed = cv2.warpPerspective(showPic, M, (w, h))

    cv2.imshow("processed", processed)
    cv2.imwrite("../result/screen/processed2.jpg", processed)
    cv2.waitKey()
    return hull, processed


def template_match():
    template = cv2.imread('weibo_ipad.png', 1)
    target = cv2.imread('images/1.jpg', 1)
    # target = cv.pyrDown(target)
    # target = cv.pyrDown(target)
    cv2.imshow('template image', template)
    cv2.imshow('target image', target)
    # cv.imwrite('images/test.jpg', target)
    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    th, tw = template.shape[:2]
    for md in methods:
        print(md)
        result = cv2.matchTemplate(target, template, md)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if md == cv2.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv2.rectangle(target, tl, br, (0, 0, 255), 2)
        cv2.imshow('match' + np.str(md), target)
    return tl, br


if __name__ == "__main__":
    # 读入图片
    # img = cv2.imread('../img/IMG_2961.JPG')
    # img = cv2.imread('../img/IMG_4218.JPG')
    # IMG = cv2.imread('../img/IMG_4238.JPG')
    # img = cv2.imread('../img/IMG_4380.JPG')
    # img = cv2.imread('../img/a.jpg')
    IMG = cv2.imread('../img/b.jpg')
    # IMG = cv2.imread('../img/c.jpg')
    # img = cv2.imread('../img/pic.png')
    find_3d_screen_image_region(IMG)
