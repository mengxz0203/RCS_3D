from config import FilePath


class GetPath(object):
    """
    def __init__(self,
                 template_image,
                 screen_image,
                 ):
        self.template_image = template_image
        self.screen_image = screen_image
        # self.template_image_path = self.get_template_path(template_image)
        # self.screen_image_path = self.get_screen_path(screen_image)
    """
    # 获取模板图片路径
    def get_template_path(self, template_image):
        template_dir = FilePath.template_image_dir(self)
        self.match_exist_image(template_dir, template_image)
        template_path = '{}/{}'.format(template_dir, template_image)
        print(template_path)
        return template_path

    # 获取屏幕区域图片路径
    def get_screen_path(self, screen_image):
        screen_dir = FilePath.screen_image_dir(self)
        self.match_exist_image(screen_dir, screen_image)
        screen_path = '{}/{}'.format(screen_dir, screen_image)
        print(screen_path)
        return screen_path

    # 获取原始图片路径
    def get_origin_path(self, origin_image):
        origin_dir = FilePath.origin_image_dir(self)
        self.match_exist_image(origin_dir, origin_image)
        origin_path = '{}/{}'.format(origin_dir, origin_image)
        print(origin_path)
        return origin_path

    # 获取左相机图片
    def get_left_path(self, left_image):
        left_dir = FilePath.chessboard_image_left_dir(self)
        self.match_exist_image(left_dir, left_image)
        left_path = '{}/{}'.format(left_dir, left_image)
        print(left_path)
        return left_path

    # 获取右相机图片
    def get_right_path(self, right_image):
        right_dir = FilePath.chessboard_image_right_dir(self)
        self.match_exist_image(right_dir, right_image)
        right_path = '{}/{}'.format(right_dir, right_image)
        print(right_path)
        return right_path

    # 检测图片是否存在
    def match_exist_image(self, dir_path, image):
        try:
            open(dir_path + '/' + image)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)


# 测试
img = GetPath()

img.get_screen_path('1.jpg')
img.get_origin_path('b.jpg')



