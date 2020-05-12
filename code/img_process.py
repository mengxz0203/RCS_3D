import cv2


def show_img(img_name):
    cv2.imshow(img_name + "window", img_name)
    cv2.waitKey()
