from ExtractCharsByColor import get_chars

state = 'IL'

for i in range(1, 19):
    if i <= 9:
        file_name = state + "0" + str(i) + ".png"
    else:
        file_name = state + str(i) + ".png"
    get_chars(file_name)



