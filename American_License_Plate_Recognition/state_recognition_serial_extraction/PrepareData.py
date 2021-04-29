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


state2index = {
    'AZ': 0, 'CA': 1, 'CO': 2, 'FL': 3, 'GA': 4, 'IL': 5, 'MA': 6, 'MD': 7, 'MI': 8, 'MO': 9,
    'NC': 10, 'NJ': 11, 'NY': 12, 'OH': 13, 'PA': 14, 'TN': 15, 'TX': 16, 'VA': 17, 'WA': 18, 'WI': 19,
}

states = [
    'AZ', 'CA', 'CO', 'FL', 'GA', 'IL', 'MA', 'MD', 'MI', 'MO',
    'NC', 'NJ', 'NY', 'OH', 'PA', 'TN', 'TX', 'VA', 'WA', 'WI',
]


def store_img_in_array():
    train_array = []
    test_array = []
    img_list = get_file_list('../data/state_feature', [], 'png')

    for dec in range(int(len(img_list) / 10)):
        go_to_test1 = int(np.random.uniform(0, 10))
        go_to_test2 = go_to_test1 + 1
        if go_to_test2 == 10:
            go_to_test2 = 1

        for sin in range(10):
            index = dec * 10 + sin
            img_path = img_list[index]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            state_index = state2index[img_name[0:2]]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_tensor = torch.tensor(img, dtype=torch.int)

            img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=True)

            if sin == go_to_test1 or sin == go_to_test2:
                test_array.append([img_tensor, state_index])
            else:
                train_array.append([img_tensor, state_index])

    print('Get {} training images'.format(len(train_array)))
    print('Get {} test images'.format(len(test_array)))

    return train_array, test_array


# train_data, test_data = store_img_in_array()

