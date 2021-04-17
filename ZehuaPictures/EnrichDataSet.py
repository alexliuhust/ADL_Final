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


def add_noise(img):
    count = int(np.random.uniform(0, 800))
    for point in range(count):
        xi = int(np.random.uniform(0, img.shape[1]))
        xj = int(np.random.uniform(0, img.shape[0]))
        img[xj, xi, 0] = 25
        img[xj, xi, 1] = 20
        img[xj, xi, 2] = 20


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
    newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))

    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2

    return cv2.warpAffine(image, M, (newW, newH))


def enrich_set(img_list, n=1):
    for x in range(n):
        img_path = img_list[x]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (318, 159), interpolation=cv2.INTER_AREA)

        count = 0
        for i in range(4):
            cut_img = img[i:150 + i, 3 * i:300 + 3 * i]
            cv2.imwrite('./data/enriched/' + img_name + str(count) + '.png', cut_img)
            count += 1

            for j in range(4):
                if j <= 1:
                    noise_img = cut_img.copy()
                    add_noise(noise_img)
                    cv2.imwrite('./data/enriched/' + img_name + str(count) + '.png', noise_img)
                    count += 1

                elif j >= 2:
                    rotate_img = cut_img.copy()
                    angle = np.random.uniform(-2.0, 2.0)
                    rotate_img = rotate_bound(rotate_img, angle)
                    rotate_img = cv2.resize(rotate_img, (300, 150), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('./data/enriched/' + img_name + str(count) + '.png', rotate_img)
                    count += 1
        if (x + 1) % 10 == 0:
            print(str(x) + " origin images have been enriched")


def do_enrichment():
    org_img_folder = './data/origin'
    img_list = get_file_list(org_img_folder, [], 'png')
    enrich_set(img_list, len(img_list))


# do_enrichment()
