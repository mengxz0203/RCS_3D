import cv2
import numpy as np
import math
from image_process import show_image
from PIL import Image

MIN_MATCH_COUNT = 5  # 最低匹配点数
MAX_LOOP_COUNT = 5  # 最大循环次数


class TemplateMatchResult(object):

    # similarity 为模板匹配算法识别相似度
    # x,y 为模板匹配算法识别后最终结果
    # h,w 为模板匹配区域长宽，可忽略
    def __init__(self, similarity, x, y, h=0, w=0):
        self.similarity = similarity
        self.x = x
        self.y = y
        self.h = h
        self.w = w

    # 返回记录的相似度
    def get_similarity(self):
        return self.similarity

    # 返回记录的匹配位置
    def get_coordinates(self):
        return self.x, self.y

    # 返回匹配区域像素大小
    def get_image_size(self):
        return self.h, self.w

    # 更新数据
    def set_data(self, similarity, x, y):
        self.similarity = similarity
        self.x = x
        self.y = y

    # 坐标变换
    def add_coordinates(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy


# BBS算法根据模板图像计算不同的权重
def patch_size(size):
    area = size[0] * size[1]
    if area < 2000:
        return 3

    elif area < 8000:
        return 5

    else:
        return 7


def read_template(template_image_path):
    template = cv2.imread(template_image_path)
    show_image(template)
    return template


def read_target(screen_image_path):
    screen = cv2.imread(screen_image_path)
    # target = cv.pyrDown(target)
    # target = cv.pyrDown(target)
    show_image(screen)
    return screen



# 分块直方图找相似位置
def block_difference(hist1, hist2):
    similarity = 0

    for i in range(len(hist1)):
        if hist1[i] == hist2[i]:
            similarity += 1
        else:
            similarity += 1 - float(abs(hist1[i] - hist2[i])) / max(hist1[i], hist2[i])

    return similarity / len(hist1)


# 分块直方图发计算匹配相似度
def get_bhmatch_similarity():
    target = read_target()
    template = read_template()

    # 将两张图缩放到统一尺寸
    image1 = target.resize((64, 64)).convert('RGB')
    image2 = template.resize((64, 64)).convert('RGB')
    # 分块直方图法
    similarity = 0;
    for i in range(4):
        for j in range(4):
            hist1 = image1.crop((i * 16, j * 16, i * 16 + 15, j * 16 + 15)).copy().histogram()
            hist2 = image2.crop((i * 16, j * 16, i * 16 + 15, j * 16 + 15)).copy().histogram()
            similarity += __block_difference(hist1, hist2)

    return similarity / 16


def bh_match_loop(screen_image, widget_image, loop_num):
    target = read_target()
    template = read_template()
    # 模板匹配循环，原图、控件、循环次数
    # 根据模板匹配算法寻找图中是否含有小图片，loop_num代表匹配循环测试
    target_size = target.size
    template_match_result = TemplateMatchResult(0, 0, 0)

    # 截取的图片距原图片左边界距离x， 距上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h
    x, y = 0, 0
    [w, h] = template.size

    # 将切割图片向周围移动距离比例
    # 循环次数越多，切分越详细
    proportion = 5 * loop_num
    foot_x, foot_y = 0, 0

    # 向左右及上下移动次数
    foot_x_all = math.ceil((target_size[0] - w) * proportion / w)
    foot_y_all = math.ceil((target_size[1] - h) * proportion / h)

    # 开始地毯式匹配相似区域
    for foot_x in range(foot_x_all + 1):
        y = 0
        for foot_y in range(foot_y_all + 1):

            # 剪切和widget一样大小的图片
            screen_region = screen_image.crop((x, y, x + w, y + h))
            result = __get_bhmatch_similarity(screen_region, widget_image)  # 两张同等大小的图像相似度
            if result > template_match_result.get_similarity():
                """寻找最相似的点"""
                template_match_result.save_data(result, x, y)

            y = y + h / proportion
            if y + h > target_size[1]:
                y = target_size[1] - h

        x = x + w / proportion
        if x + w > target_size[0]:
            x = target_size[0] - w

    # 每次循环都会提高阈值
    if template_match_result.get_similarity() < 1 - loop_num / 100:
        # 超出循环次数则结束
        if loop_num > MAX_LOOP_COUNT:
            return template_match_result

        else:
            x, y = template_match_result.get_coordinate()
            # 缩小匹配范围
            if x > 0:
                x = x - w / proportion
                if y > 0:
                    y = y - h / proportion

                length = min(x + w + 2 * w / proportion, target_size[0])
                height = min(y + h + 2 * h / proportion, target_size[1])

                # 将下一次查找的区域剪切出来
                screen_region = screen_image.crop((x, y, length, height))
                region_tmr = __bh_match_loop(screen_region, widget_image, loop_num + 1)

                # 递归查找模板匹配最符合要求的区域
                if template_match_result.get_similarity() > region_tmr.get_similarity():
                    template_match_result.add_coordinate(w / 2, h / 2)

                else:
                    region_x, region_y = region_tmr.get_coordinate()
                    template_match_result.save_data(region_tmr.get_similarity, x + region_x, y + region_y)

                return template_match_result

    # 比阈值小或者匹配范围已经最小则返回记录的最相似区域
    template_match_result.add_coordinate(w / 2, h / 2)
    return template_match_result


# opencv模板匹配
def opencv_template_match(algorithm):
    target = read_target()
    template = read_template()
    th, tw = template.shape[:2]
    result = cv2.matchTemplate(target, template, algorithm)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if algorithm == cv2.TM_SQDIFF_NORMED:
        tl = min_loc
    else:
        tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    cv2.rectangle(target, tl, br, (0, 0, 255), 2)
    show_image(target)
    cv2.destroyAllWindows()


# 相关系数匹配法
def tm_ccoeff_match():
    return opencv_template_match(cv2.TM_CCOEFF)


# 归一化相关系数匹配法
def tm_ccoeff_normed_match():
    return opencv_template_match(cv2.TM_CCOEFF_NORMED)


# 相关匹配
def tm_ccorr_match():
    return opencv_template_match(cv2.TM_CCORR)


# 归一化相关匹配
def tm_ccorr_normed_match():
    return opencv_template_match(cv2.TM_CCORR_NORMED)


# 平方差匹配
def tm_sqdiff_match():
    return opencv_template_match(cv2.TM_SQDIFF)


# 归一化平方差匹配
def tm_sqdiff_normed_match():
    return opencv_template_match(cv2.TM_SQDIFF_NORMED)


# 基于FLANN的匹配器(FLANN based Matcher)定位图片
def flann_based_match(algorithm):
    template = read_template()
    target = read_target()

    # 找出图像中关键点
    kp1, des1 = algorithm.detectAndCompute(template, None)
    kp2, des2 = algorithm.detectAndCompute(target, None)

    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # KTreeIndex配置索引，指定待处理核密度树的数量
    search_params = dict(checks=60)  # 指定递归遍历的次数。值越高结果越准确
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)  # 进行匹配

    # 存储所有符合比率测试的匹配项
    good_dot = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            """
            ratio=0. 4：对于准确度要求高的匹配； 
            ratio=0. 6：对于匹配点数目要求比较多的匹配；
            ratio=0. 5：一般情况下。
            """
            good_dot.append(m)

    if len(good_dot) > MIN_MATCH_COUNT:
        # trainIdx    是匹配之后所对应关键点的序号，大图片的匹配关键点序号
        screen_pts = np.float32([kp2[m.trainIdx].pt for m in good_dot]).reshape(-1, 1, 2)
        # queryIdx  是匹配之后所对应关键点的序号，控件图片的匹配关键点序号
        widget_pts = np.float32([kp1[m.queryIdx].pt for m in good_dot]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(widget_pts, screen_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
        x, y = [], []
        for i in range(len(dst)):
            x.append(int(dst[i][0][0]))
            y.append(int(dst[i][0][1]))

        template_match_result = TemplateMatchResult(1, min(x), min(y), max(x) - min(x), max(y) - min(y))
        return template_match_result

    else:
        print("Not enough matches are found - %d/%d" % (len(good_dot), MIN_MATCH_COUNT))
        matchesMask = None
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        result = cv2.drawMatches(template, kp1, target, kp2, good_dot, None, **draw_params)
        show_image(result)
        template_match_result = TemplateMatchResult(0, 0, 0, 0, 0)
        return template_match_result


# 基于FlannBasedMatcher的SIFT实现模板匹配
def sift_match():
    # 实例化sift
    sift = cv2.xfeatures2d.SIFT_create()
    return flann_based_match(sift)


# 基于FlannBasedMatcher的SURF实现
def surf_match():
    # 实例化surf
    surf = cv2.xfeatures2d.SURF_create(400)
    return flann_based_match(surf)


# if __name__ == '__main__':
#     tm_ccoeff_match()

TEMPLATE_MATCHERS = {

            'tcf': tm_ccoeff_match,
            'tcfn': tm_ccoeff_normed_match,
            'tcr': tm_ccorr_match,
            'tcrn': tm_ccorr_normed_match,
            'ts': tm_sqdiff_match,
            'tsn': tm_sqdiff_normed_match,
            'sift': sift_match,
            'surf': surf_match
}
