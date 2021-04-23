import cv2
from matplotlib import pyplot as plt

'''性能很好，但是目前无法处理字符粘连的情况'''


def segment_character(input_image):  # path参数为读取图片的路径
    # img_original = cv2.imread(path)  # 读取图片
    # img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # 转换了灰度化

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 获得膨胀的kernel
    image = cv2.dilate(input_image, kernel)  # 膨胀处理

    # 查找轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)  # boundingRect返回分别为x，y，w，h：x，y是矩形左上点的坐标，w，h是矩形的宽和高
        x = rect[0]  # 矩形左上点的横坐标
        y = rect[1]  # 矩形左上点的纵坐标
        width = rect[2]  # 矩形的宽
        height = rect[3]  # 矩形的高
        word.append(x)  # word[0]
        word.append(y)  # word[1]
        word.append(width)  # word[2]
        word.append(height)  # word[3]
        words.append(word)
    words = sorted(words, key=lambda s: s[0], reverse=False)  # 按矩形左上点的横坐标将words排序
    i = 0
    for word in words:
        # 根据轮廓的外接矩形筛选轮廓
        if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10) and \
                (word[3] > 0.5 * input_image.shape[0]) and (word[2] > 0.05 * input_image.shape[1]):
            i = i + 1
            split_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]  # 分割原图片成字符图像
            split_image = cv2.resize(split_image, (25, 40))  # 统一字符图像的尺寸
            word_images.append(split_image)


    for i, j in enumerate(word_images):
        plt.subplot(1, 8, i + 1)  # 暂定最大8位字符
        plt.imshow(word_images[i], cmap='gray')  # 画灰度图
    plt.show()

    return i + 1  # 返回字符数目


# 测试
# num = segment_character('../Character_segmentation_image/18.PNG')
