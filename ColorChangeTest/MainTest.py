from ExtractCharsByColor import get_chars
import cv2

state = 'TN'

for i in range(1, 19):
    if i <= 9:
        file_name = state + "0" + str(i) + ".png"
    else:
        file_name = state + str(i) + ".png"

    get_chars(file_name)
    # cv2.imwrite('./Extracted/' + file_name, extr)

