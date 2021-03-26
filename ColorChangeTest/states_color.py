
def get_color_range(state):
    s = state[0:2]
    if s == "MA":
        return [[1, 165], [1, 145], [130, 240]]
    elif s == "MD" or state == "NE":
        # return [[25, 38], [25, 38], [25, 38]]
        return [[0, 38], [0, 38], [0, 38]]
    elif s == "MS":
        return [[38, 130], [21, 78], [10, 60]]

