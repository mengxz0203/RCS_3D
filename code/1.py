import cv2
import numpy as np


def read_template():
    template = cv2.imread('../images/template/weibo_ipad.png', 1)
    cv2.imshow('template image', template)
    # cv2.waitKey(0)
    return template

def read_target():
    target = cv2.imread('images/1.jpg', 1)
    # target = cv.pyrDown(target)
    # target = cv.pyrDown(target)
    # cv2.imshow('target image', target)
    # cv2.waitKey(0)
    # cv.imwrite('images/test.jpg', target)
    return target


def show():
    th, tw = read_template().shape[:2]
    result = cv2.matchTemplate(read_target(), read_template(), cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    cv2.rectangle(read_target(), tl, br, (0, 0, 255), 2)
    cv2.imshow('match', read_target())


show()