import cv2
from matplotlib import pyplot as plt


def segment_character(input_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(input_image, kernel)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        word.append(x)  # word[0]
        word.append(y)  # word[1]
        word.append(width)  # word[2]
        word.append(height)  # word[3]
        words.append(word)
    words = sorted(words, key=lambda s: s[0], reverse=False)
    i = 0
    for word in words:

        if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 5)) and (word[2] > 10) and \
                (word[3] > 0.5 * input_image.shape[0]) and (word[2] > 0.01 * input_image.shape[1]):
            i = i + 1
            split_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            split_image = cv2.resize(split_image, (25, 40))
            split_image = cv2.copyMakeBorder(split_image, 3, 3, 15, 14, cv2.BORDER_CONSTANT)
            word_images.append(split_image)

    for i, j in enumerate(word_images):
        plt.subplot(1, 8, i + 1)
        plt.imshow(word_images[i], cmap='gray')
    plt.show()

    return word_images
