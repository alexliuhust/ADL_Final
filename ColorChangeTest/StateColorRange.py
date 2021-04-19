ranges = {
    "MA": [[1, 165], [1, 145], [130, 240]],
    "MD": [[0, 122], [0, 81], [0, 65]],
    "GA": [[0, 122], [0, 81], [0, 65]],
    "NJ": [[0, 93], [0, 88], [0, 82]],
}


def get_color_range(file_name):
    state = file_name[0:2]
    return ranges[state]
