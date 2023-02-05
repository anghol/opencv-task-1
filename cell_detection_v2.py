import cv2 as cv
import utils
import numpy as np
from consts import *

def get_cells(image: cv.Mat, algorithm: str) -> cv.Mat:
    if algorithm == Mask_Alg.BLOB_DETECT:
        image_detected_cells, mask, keypoints = detect_ellipses_alg(image)
    return mask, image_detected_cells

def detect_ellipses_alg(image: cv.Mat):
    # preprocessing: blur and grayscale
    image_blur = utils.filter_image(image, Filter.AVERAGE, average_ksize=12)
    image_gray = utils.rgb_to_gray(image_blur)

    # preprocessing: threshold to get binary inverted image and closing
    image_binary = cv.threshold(image_gray, Color.BLACK.value, Color.WHITE.value,
                            cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    image_binary = utils.morph_transform(image_binary, Morph.CLOSING, ksize=6)

    params = cv.SimpleBlobDetector_Params()

    params.collectContours = True

    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 1000000

    params.filterByCircularity = True 
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.5

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_binary)
    image_detected_cells = cv.drawKeypoints(image_gray, keypoints, np.array([[0]]),
                            (255, 0, 0))

    contours = detector.getBlobContours()
    mask = np.zeros(image_binary.shape, 'uint8')
    for contour in contours:
        if cv.contourArea(contour) / (image_binary.shape[0] * image_binary.shape[1]) > 0.5:
            contour_mask = np.zeros(image_binary.shape, 'uint8') + 255
            cv.drawContours(contour_mask, [contour], -1, (0, 0, 0), -1)
        else:
            contour_mask = np.zeros(image_binary.shape, 'uint8')
            cv.drawContours(contour_mask, [contour], -1, (255, 255, 255), -1)
        mask = cv.bitwise_or(mask, contour_mask)

    return image_detected_cells, mask, keypoints