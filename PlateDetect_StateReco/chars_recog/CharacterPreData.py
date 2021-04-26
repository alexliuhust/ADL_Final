import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable


def get_file_list(dir, file_list, ext=None):
    new_dir = dir
    if os.path.isfile(dir):
        if ext is None:
            file_list.append(dir)
        else:
            if ext in dir[-3:]:
                file_list.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir, s)
            get_file_list(new_dir, file_list, ext)

    return file_list


classes2index = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29,
    'V': 30, 'W': 31, 'X': 32, 'Y': 33, 'Z': 34,
}

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
           'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')


def store_img_in_array():
    train_array = []
    test_array = []
    img_list = get_file_list('./new_datasource', [], 'png')

    for dec in range(int(len(img_list) / 10)):
        go_to_test1 = int(np.random.uniform(0, 10))
        go_to_test2 = go_to_test1 + 1
        if go_to_test2 == 10:
            go_to_test2 = 1

        for sin in range(10):
            index = dec * 10 + sin
            img_path = img_list[index]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            class_index = classes2index[img_name[0:1]]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_tensor = torch.tensor(img, dtype=torch.int)

            img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=True)

            if sin == go_to_test1 or sin == go_to_test2:
                test_array.append([img_tensor, class_index])
            else:
                train_array.append([img_tensor, class_index])

    print('Get {} training images'.format(len(train_array)))
    print('Get {} test images'.format(len(test_array)))

    return train_array, test_array


# train_data, test_data = store_img_in_array()

