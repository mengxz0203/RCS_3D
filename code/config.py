import os
from os import path


class FilePath(object):

    # 模板图片绝对路径
    def template_image_dir(self):
        image_dir = path.join(os.getcwd(), 'template/')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def screen_image_dir(self):
        image_dir = path.join(os.getcwd(), 'screen/')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir