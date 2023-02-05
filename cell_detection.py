import cv2 as cv
import utils
import numpy as np
from consts import *
from matplotlib import pyplot as plt


def get_cells(image: cv.Mat, algorithm: str) -> cv.Mat:
    if algorithm == Mask_Alg.BLOB_DETECT:
        image_detected_cells, mask, keypoints = detect_ellipses_alg(image)
    return mask, image_detected_cells

def detect_ellipses_alg(image: cv.Mat):
    # preprocessing: blur and grayscale
    image_blur = utils.filter_image(image, Filter.AVERAGE, average_ksize=12)
    image_gray = utils.rgb_to_gray(image_blur)

    # preprocessing: threshold to get binary inverted image and closing
    ret, image_binary = cv.threshold(image_gray, Color.BLACK.value, Color.WHITE.value,
                            cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    image_binary = utils.morph_transform(image_binary, Morph.CLOSING)

    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 1000000

    params.filterByCircularity = True 
    params.minCircularity = 0.2
    params.maxCircularity = 1

    params.filterByConvexity = True
    params.minConvexity = 0.2
    params.maxConvexity = 1

    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.5

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_binary)

    mask = np.zeros(image_binary.shape, 'uint8') + Color.WHITE.value

    for keypoint in keypoints:
        x_key, y_key = [round(coord) for coord in keypoint.pt]
        size = round(keypoint.size)
        start_x, stop_x = x_key - size, x_key + size + 1
        start_y, stop_y = y_key - size, y_key + size + 1
        mask[start_y:stop_y, start_x:stop_x] = image_binary[start_y:stop_y, start_x:stop_x]
    mask = 255 - mask

    cnts = np.zeros(image_binary.shape)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 500]
    cv.drawContours(cnts, contours, -1, (255, 255, 255), -1)
    mask = cnts.astype('uint8')

    image_detected_cells = cv.drawKeypoints(image_gray, keypoints, np.array([[0]]),
                            (255, 0, 0))
    cv.drawContours(image_detected_cells, contours, -1, (0, 255, 255), 3)

    return image_detected_cells, mask, keypoints