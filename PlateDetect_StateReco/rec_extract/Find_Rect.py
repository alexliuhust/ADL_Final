import cv2
import numpy as np


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_rect(img, rect):  # https://blog.csdn.net/loovelj/article/details/90080725
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def find_rect(img_path):
    im = cv2.imread(img_path)
    car_img = im.copy()
    ratio = im.shape[1] / im.shape[0]
    car_img = cv2.resize(car_img, (300, int(300 / ratio)), interpolation=cv2.INTER_AREA)
    cv_show(car_img, 'car')

    imgray = im.copy()
    if len(im.shape) == 3:
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = contours[0]

    img_return = []

    i = 1

    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)

        width = rect[1][0]
        height = rect[1][1]
        if width == 0 or height == 0:
            continue
        if width / height < 0.2:
            continue
        if (height < 0.05 * im.shape[0]) or (width < 0.05 * im.shape[1]):
            continue
        if (height > 0.8 * im.shape[0]) or (width > 0.8 * im.shape[1]):
            continue

        img_crop, img_rot = crop_rect(im, rect)
        img_return.append(img_crop)

    return img_return

