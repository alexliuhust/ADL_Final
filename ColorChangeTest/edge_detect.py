import cv2
import numpy as np


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("./" + "MD" + "test.png")
cv_show(img, "Origin")

height = img.shape[0]
width = img.shape[1]
h_c = int(height / 6)
w_c = int(width / 20)
img = img[h_c:height - h_c, w_c:width - w_c]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(img, "Cut")


sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
img = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
cv_show(img, "Edge")
