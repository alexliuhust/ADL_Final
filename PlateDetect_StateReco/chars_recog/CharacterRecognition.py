from chars_recog.CharacterNetTrain import load_test


def get_char_recog_result(images):
    str1 = [''] * len(images)
    for i in range(len(images)):
        str1[i] = load_test(images[i])

    result = ''
    for i in range(len(images)):
        result = result + str1[i]

    return result