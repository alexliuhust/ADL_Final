import os
import cv2
import numpy as np


def get_file_list(dir, file_list, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            file_list.append(dir)
        else:
            if ext in dir[-3:]:
                file_list.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_file_list(newDir, file_list, ext)

    return file_list


org_img_folder = './data/origin'

img_list = get_file_list(org_img_folder, [], 'png')


def add_noise(img):
    count = int(np.random.uniform(0, 1000))
    for point in range(count):
        xi = int(np.random.uniform(0, img.shape[1]))
        xj = int(np.random.uniform(0, img.shape[0]))
        img[xj, xi, 0] = 25
        img[xj, xi, 1] = 20
        img[xj, xi, 2] = 20


for x in range(1):
    img_path = img_list[x]
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_AREA)

    for i in range(11):
        cut_img = img.copy()
        cut_img = cut_img[i:150+i, 2*i:300+2*i]

        to_add_noise = int(np.random.uniform(0, 8))
        if 0 <= to_add_noise <= 1:
            add_noise(cut_img)

        save_path = './data/enriched/' + img_name + str(i) + '.png'
        cv2.imwrite(save_path, cut_img)





