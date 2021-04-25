from rec_extract.FindRect_And_Determine import findRect_and_determine
from state_recog_char_extract.StateRecogResult import get_state_name
from state_recog_char_extract.ExtractCharsByColor import get_chars
from chars_fregment.Segment_Character_Function import segment_character


def do_big_test(image_path):
    plate_image = findRect_and_determine(image_path)
    state_name = get_state_name(plate_image)
    extracted_image = get_chars(plate_image, state_name)
    word_images, n = segment_character(extracted_image)


# do_big_test('data/image_license_plate/01.png')
# do_big_test('data/image_license_plate/02.png')
# do_big_test('data/image_license_plate/03.png')
# do_big_test('data/image_license_plate/04.png')
# do_big_test('data/image_license_plate/05.png')
# do_big_test('data/image_license_plate/06.png')
# do_big_test('data/image_license_plate/07.png')
# do_big_test('data/image_license_plate/08.png')




