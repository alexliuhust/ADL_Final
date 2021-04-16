import cv2


def get_top_bottom(img):
    height = img.shape[0]
    h_c = int(height / 4)
    top = img[0:h_c, :]
    bottom = img[height - h_c: height, :]
    out = cv2.vconcat([top, bottom])

    return out


def get_edge(img, top_bottom):
    if top_bottom:
        img = get_top_bottom(img)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    edge = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    for i in range(0, edge.shape[0]):
        for j in range(0, edge.shape[1]):
            data = (edge[i, j])
            if data >= 120:
                edge[i, j] = 255
            else:
                edge[i, j] = 0
    return edge
