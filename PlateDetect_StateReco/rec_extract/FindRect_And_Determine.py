import cv2
from PIL import Image
import numpy as np
from rec_extract.Find_Rect import find_rect
from rec_extract.TFCNNresult_ import *


def findRect_and_determine(image_path):
    rect_images = find_rect(image_path)  # rect_images是一个list，存储所有的矩形图片

    if(len(rect_images) < 1):  # rect_images为空，即没有矩形，抛异常
        raise RuntimeError('Cannot detect any rectangle.')

    num_True = 0  # 判断为True的个数
    image_return = rect_images[0]  # 为返回值开辟空间

    for image in rect_images:
        result = single_test_by_dataflow(image)  # 将矩形图片传入第一个网络做True or False判断

        if (result == 1):  # True
            num_True = num_True + 1  # num_True 加1
            image_return = image

            # 展示True图片
            # cv2.imshow('1', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # num_True不等于1时，抛异常
    if(num_True > 1):
        raise RuntimeError('The num of True image in the 1st CNN is more than one.')
    if (num_True == 0):
        raise RuntimeError('Cannot detect any True image in the 1st CNN.')

    return image_return


# test
# image = findRect_and_determine('./image_license_plate/17.jpeg')
# cv2.imshow('True', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
