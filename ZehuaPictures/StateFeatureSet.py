import cv2
import os
from EnrichDataSet import get_file_list
from GetFeature import get_edge


def get_feature(img_list, n=1):
    for x in range(n):
        img_path = img_list[x]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        edge = get_edge(img, True)

        cv2.imwrite('./data/state_feature/' + img_name + str(x) + '.png', edge)

        if (x % 500) == 0:
            print(str(x) + " features have been made")


def do_get_feature():
    org_img_folder = '../../TFdata/True'
    img_list = get_file_list(org_img_folder, [], 'png')
    get_feature(img_list, len(img_list))


# do_get_feature()
