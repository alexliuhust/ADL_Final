
def get_color_range(state):
    if state == "MA":
        return [[1, 55], [1, 90], [130, 170]]
    elif state == "MD" or state == "NE":
        # return [[25, 38], [25, 38], [25, 38]]
        return [[0, 38], [0, 38], [0, 38]]
    elif state == "MS":
        return [[90, 120], [40, 70], [30, 60]]

