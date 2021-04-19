
def get_color_range(file_name):
    state = file_name[0:2]
    if state == "MA":
        return [[1, 165], [1, 145], [130, 240]]
    elif state == "MD" or state == "NE":
        return [[0, 38], [0, 38], [0, 38]]
    elif state == "MS":
        return [[38, 130], [21, 78], [10, 60]]

