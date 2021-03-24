import cv2
from PIL import Image
import matplotlib.pyplot as plt


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('./MAtest01.png')
print(img.shape)
cv_show(img, "origin")

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

cv_show(img, "color_change")


# img = Image.open("./MAtest01.png")
# img_origin = img
# print(img_origin.size)
# plt.imshow(img_origin)
# plt.show()
#
# width = img.size[0]
# height = img.size[1]
#
# color_range = [[130, 170], [1, 90], [1, 55]]
#
# for i in range(0, width):
#     for j in range(0, height):
#         data = (img.getpixel((i, j)))
#
#         if color_range[0][0] <= data[0] <= color_range[0][1] \
#                 and color_range[1][0] <= data[1] <= color_range[1][1] \
#                 and color_range[2][0] <= data[2] <= color_range[2][1]:
#             img.putpixel((i, j), (255, 255, 255))
#         else:
#             img.putpixel((i, j), (0, 0, 0))
#
# cv_show(img)
#
# img = img.convert("LA")
# plt.imshow(img)
# plt.show()


