from chars_segmentation_recognition.CharacterNetTrain import load_test


def get_char_recog_result(images):
    str1 = [''] * len(images)
    for i in range(len(images)):
        str1[i] = load_test(images[i])

    result = ''
    for i in range(len(images)):
        if str1[i] == '#':
            continue
        result = result + str1[i]

    return result
