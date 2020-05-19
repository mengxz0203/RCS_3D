import os
from os import path

root = path.join(path.dirname(path.dirname(__file__)))
print(root)


class FilePath(object):

    # 模板图片绝对路径
    def template_image_dir(self):
        image_dir = path.join(root, 'images/template/')
        print("path:", image_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    # 屏幕区域图片绝对路径
    def screen_image_dir(self):
        image_dir = path.join(root, 'images/screen/')
        print("path:", image_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    # 原始图片路径
    def origin_image_dir(self):
        image_dir = path.join(root, 'images/origin')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    # 标定图片（左）
    def chessboard_image_left_dir(self):
        image_dir = path.join(root, 'images/left')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    # 标定图片（右）
    def chessboard_image_right_dir(self):
        image_dir = path.join(root, 'images/right')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir


# 测试
# FilePath.template_image_dir()
