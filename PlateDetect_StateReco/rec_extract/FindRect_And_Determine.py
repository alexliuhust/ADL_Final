from rec_extract.Find_Rect import find_rect
from rec_extract.TFCNNresult_ import *


def findRect_and_determine(image_path):
    rect_images = find_rect(image_path)

    if len(rect_images) < 1:
        raise RuntimeError('Cannot detect any rectangle.')

    num_True = 0
    image_return = rect_images[0]

    for image in rect_images:
        result = single_test_by_dataflow(image)

        if result == 1:  # True
            num_True = num_True + 1
            image_return = image

    if num_True > 1:
        raise RuntimeError('The num of True image in the 1st CNN is more than one.')
    if num_True == 0:
        raise RuntimeError('Cannot detect any True image in the 1st CNN.')

    return image_return
