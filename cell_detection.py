import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt

def get_coordinate_grid(height, width):
    coord_x_line = np.arange(width).reshape((1, width))
    coords_x = np.repeat(coord_x_line, height, axis=0)

    coord_y_line = np.arange(height).reshape((1, height))
    coords_y = np.repeat(coord_y_line, width, axis=0).T

    return coords_x, coords_y

def get_cells(image: cv2.Mat, algorithm: str) -> cv2.Mat:
    if algorithm == 'Ellipse detect':
        image_detected_cells, mask, keypoints = detect_ellipses_alg(image)
    return mask, image_detected_cells

def detect_ellipses_alg(image: cv2.Mat):
    # preprocessing: blur and grayscale
    image_blur = utils.filter_image(image, 'Average', average_ksize=12)
    image_gray = utils.rgb_to_gray(image_blur)

    # preprocessing: threshold to get binary inverted image and closing
    ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    image_binary = utils.morph_transform(image_binary, 'closing')

    params = cv2.SimpleBlobDetector_Params()

    params.minDistBetweenBlobs = 5

    params.filterByArea = True
    params.minArea = 2000
    params.maxArea = 400000

    params.filterByCircularity = True 
    params.minCircularity = 0.2

    params.filterByConvexity = True
    params.minConvexity = 0.2

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_binary)

    height, width = image.shape
    mask, coords = np.zeros((height, width), 'int'), np.zeros((height, width), 'int')
    coords_x, coords_y = get_coordinate_grid(height, width)

    for keypoint in keypoints:
        x_key, y_key = keypoint.pt[0], keypoint.pt[1]
        size = keypoint.size
        coords = ((x_key - coords_x)**2 + (y_key - coords_y)**2)**(0.5)
        mask = (mask + np.where(coords <= size , 255-image_binary, 0)) % 256
    mask = utils.morph_transform(mask.astype('uint8'), 'opening', ksize=7)

    blank = np.zeros((1, 1))
    image_detected_cells = cv2.drawKeypoints(image_binary, keypoints, blank,
                            (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image_detected_cells, mask, keypoints