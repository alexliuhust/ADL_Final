import cv2
import numpy as np
from states_color import get_color_range
from edge_detect import get_edge


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_chars(state):
    org = cv2.imread("./" + state + "test.png")
    cv_show(org, "Origin")
    input = org.copy()
    edge = get_edge(input)

    height = org.shape[0]
    width = org.shape[1]
    h_c = int(height / 5)
    w_c = int(width / 20)
    img = org[h_c:height - h_c, w_c:width - w_c]
    color_range = get_color_range(state)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            data = (img[i, j])

            if color_range[0][0] <= data[0] <= color_range[0][1] \
                    and color_range[1][0] <= data[1] <= color_range[1][1] \
                    and color_range[2][0] <= data[2] <= color_range[2][1]:
                img[i, j] = [255, 255, 255]
            else:
                img[i, j] = [0, 0, 0]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv_show(img, "Extraction")

    dilate_num = 5
    erode_num = 2

    kernel = np.ones((dilate_num, dilate_num), dtype=np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    # cv_show(img, "Dilation ")

    kernel = np.ones((erode_num, erode_num), dtype=np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    cv_show(img, "Erosion")
    cv_show(edge, "Edge")
