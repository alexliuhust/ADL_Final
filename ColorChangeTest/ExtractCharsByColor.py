import cv2
import numpy as np
from StateColorRange import get_color_range


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_chars(file_name):
    org = cv2.imread("../ZehuaPictures/data/origin/" + file_name)
    org = cv2.resize(org, (300, 150), interpolation=cv2.INTER_AREA)
    cv_show(org, "Origin")

    height = org.shape[0]
    width = org.shape[1]
    h_c = int(height / 5)
    w_c = int(width / 40)
    img = org[h_c:height - h_c, w_c:width - w_c]
    if file_name[0:2] == "CA" or file_name[0:2] == "PA":
        img = org[int(1.4*h_c):height - int(0.6*h_c), w_c:width - w_c]

    color_range = get_color_range(file_name)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            data = (img[i, j])

            if color_range[0][0] <= data[0] <= color_range[0][1] \
                    and color_range[1][0] <= data[1] <= color_range[1][1] \
                    and color_range[2][0] <= data[2] <= color_range[2][1]:
                img[i, j] = [255, 255, 255]
            else:
                img[i, j] = [0, 0, 0]

    extr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dilate_num = 2
    erode_num = 2
    for k in range(2):
        kernel = np.ones((dilate_num, dilate_num), dtype=np.uint8)
        extr = cv2.dilate(extr, kernel, iterations=2)
        kernel = np.ones((erode_num, erode_num), dtype=np.uint8)
        extr = cv2.erode(extr, kernel, iterations=1)

    cv_show(extr, file_name)

