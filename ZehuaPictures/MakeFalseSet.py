import cv2
from EnrichDataSet import get_file_list
import numpy as np


def get_random_point(ex_h, ex_w, height, width):
    point_h = int(np.random.uniform(0, height - ex_h))
    point_w = int(np.random.uniform(0, width - ex_w))
    return point_h, point_w


def cut_img(img, h, w, p_h, p_w):
    cut = img[p_h:p_h+h, p_w:p_w+w]
    return cut


def get_random_cut_img(img, cut_height, height, width):
    point_h, point_w = get_random_point(cut_height, cut_height * 2, height, width)
    cut = cut_img(img, cut_height, cut_height * 2, point_h, point_w)
    cut = cv2.resize(cut, (300, 150), interpolation=cv2.INTER_AREA)
    return cut


def make_false_pic(img_list, n=1):
    for x in range(n):
        img_path = img_list[x]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        height = img.shape[0]
        width = img.shape[1]

        for i in range(20):
            for j in range(3, 14):
                cut_height = j * 10
                cut = get_random_cut_img(img, cut_height, height, width)
                cv2.imwrite('./data/false_plate/F' + str(x) + str(i) + str(j) + '.png', cut)
            print(str((i + 1) * 200) + ' false images completed')



def do_make_false_pic():
    org_img_folder = './data/cars'
    img_list = get_file_list(org_img_folder, [], 'png')
    make_false_pic(img_list, len(img_list))


do_make_false_pic()





