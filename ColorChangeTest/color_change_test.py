from PIL import Image
import matplotlib.pyplot as plt


i = 1
j = 1
img = Image.open("./test01.png")
print(img.size)
plt.imshow(img)
plt.show()

width = img.size[0]
height = img.size[1]
for i in range(0,width):
    for j in range(0, height):
        data = (img.getpixel((i, j)))

        if data[0] >= 170 and data[1] >= 170 and data[2] >= 170:
            img.putpixel((i, j), (234, 53, 57))

img = img.convert("RGB")
plt.imshow(img)
plt.show()


