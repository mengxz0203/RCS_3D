import cv2
import numpy as np


def show_image(image_name):
    cv2.imshow('ImgWindow', image_name)
    cv2.waitKey(0)
