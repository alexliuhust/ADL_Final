import cv2
import numpy as np


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('./MAtest01.png')
print(img.shape)
cv_show(img, "Origin")

width = img.shape[0]
height = img.shape[1]

color_range = [[1, 55], [1, 90], [130, 170]]  # This is BGR system

for i in range(0, width):
    for j in range(0, height):
        data = (img[i, j])

        if color_range[0][0] <= data[0] <= color_range[0][1] \
                and color_range[1][0] <= data[1] <= color_range[1][1] \
                and color_range[2][0] <= data[2] <= color_range[2][1]:
            img[i, j] = [255, 255, 255]
        else:
            img[i, j] = [0, 0, 0]

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
cv_show(img, "Extraction")

kernel = np.ones((4, 4), dtype=np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
print(img.shape)
cv_show(img, "Dilation ")

kernel = np.ones((3, 3), dtype=np.uint8)
img = cv2.erode(img, kernel, iterations=1)
print(img.shape)
cv_show(img, "Erosion")

