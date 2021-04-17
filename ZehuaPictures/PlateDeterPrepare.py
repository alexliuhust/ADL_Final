import torch
from EnrichDataSet import get_file_list
import cv2


dataset = []


def make_dataset(org_img_folder, target):
    img_list = get_file_list(org_img_folder, [], 'png')
    for x in range(len(img_list)):
        img_path = img_list[x]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.tensor(img, dtype=torch.int)

        dataset.append([img_tensor, target])


make_dataset('./data/enriched', 1)
make_dataset('./data/false_plate', 0)





