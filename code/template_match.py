import cv2
import numpy as np


def read_template():
    template = cv2.imread('../template/qq.png', 1)
    cv2.imshow('template image', template)
    # cv2.waitKey(0)
    return template


def read_target():
    target = cv2.imread('../images/processed.jpg', 1)
    # target = cv.pyrDown(target)
    # target = cv.pyrDown(target)
    cv2.imshow('target image', target)
    # cv2.waitKey(0)
    # cv.imwrite('images/test.jpg', target)
    return target


def template_match():
    template = read_template()
    target = read_target()
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
        cv2.waitKey(0)


if __name__ == '__main__':
    template_match()
