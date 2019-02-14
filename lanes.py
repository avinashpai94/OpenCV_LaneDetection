import cv2
import numpy as np


def make_cord(image, line_params):
    slope, intr = line_params
    y1 = image.shape[0]
    y2 = int(y1*(2.5/5))
    x1 = int((y1-intr)/slope)
    x2 = int((y2-intr)/slope)
    return np.array([x1, y1, x2, y2])


def avg_slope_int(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intr = params[1]
        if slope > 0:
            right_fit.append((slope, intr))
        else:
            left_fit.append((slope, intr))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_cord(image, left_fit_avg)
    right_line = make_cord(image, right_fit_avg)
    return np.array([left_line, right_line])


def cannyify(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny = cv2.Canny(blur_img, 50, 150)
    return canny


def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_img


def region_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


capture = cv2.VideoCapture('test2.mp4')
while capture.isOpened():
    _, frame = capture.read()
    canny_img = cannyify(frame)
    canny_img = region_interest(canny_img)
    lines = cv2.HoughLinesP(canny_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    lines = avg_slope_int(frame, lines)
    line_image = display_lines(frame, lines)
    full_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', full_img)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
