import cv2


def get_edge(img):
    edge = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(edge, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(edge, cv2.CV_64F, 0, 1)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    edge = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    return edge
