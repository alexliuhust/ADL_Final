from rectangle_detection_plate_validation.FindRectAndDetermine import findRect_and_determine
from state_recognition_serial_extraction.StateRecogResult import get_state_name
from state_recognition_serial_extraction.ExtractCharsByColor import get_chars
from chars_segmentation_recognition.SegmentCharacterFunction import segment_character
from chars_segmentation_recognition.CharacterRecognition import get_char_recog_result


def overall_test(img_path):
    plate_image = findRect_and_determine(img_path)

    state_name = get_state_name(plate_image)
    extracted_image = get_chars(plate_image, state_name)

    word_images = segment_character(extracted_image)
    result = get_char_recog_result(word_images)

    print("===============================")
    print(" State: " + state_name)
    print("Serial: " + result)


path = './data/overall_test/'

for i in range(1, 18):
    if i <= 9:
        image_path = path + "0" + str(i) + ".png"
    else:
        image_path = path + str(i) + ".png"
    overall_test(image_path)
