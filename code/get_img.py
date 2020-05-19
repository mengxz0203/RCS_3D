import os
import cv2
from config import FilePath


class Image(object):
    def __init__(self,
                 template_image,
                 screen_image,
                 ):
        self.template_image_path = self.get_template_path(template_image)
        self.screen_image_path = self.get_screen_path(screen_image)

    # 获取模板图片路径
    def get_template_path(self, template_image):
        template_dir = FilePath.template_image_dir()
        self.match_exist_image((template_dir, template_image))
        template_path = '{}/{}'.format(template_dir, template_image)
        return template_path

    # 获取屏幕区域图片路径
    def get_screen_path(self, screen_image):
        screen_dir = FilePath.screen_image_dir()
        self.match_exist_image((screen_dir, screen_image))
        screen_path = '{}/{}'.format(screen_dir, screen_image)
        return screen_path

    # 检测图片是否存在
    def match_exist_image(self, dir_path, image):
        try:
            open(dir_path + '/' + image)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)

