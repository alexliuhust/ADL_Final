from rec_extract.FindRect_And_Determine import findRect_and_determine
from state_recog_char_extract.StateRecogResult import get_state_name
from state_recog_char_extract.ExtractCharsByColor import get_chars
from chars_fregment.Segment_Character_Function import segment_character

image_path = './data/image_license_plate/222.png'
plate_image = findRect_and_determine(image_path)
state_name = get_state_name(plate_image)
extracted_image = get_chars(plate_image, state_name)
segment_character(extracted_image)

