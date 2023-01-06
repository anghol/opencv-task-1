import cv2
import numpy as np
from matplotlib import pyplot as plt


def delete_description(image: cv2.Mat, size: int) -> cv2.Mat:
    height = image.shape[0]
    image_with_deleted_description = image[:height-size, :]
    return image_with_deleted_description

def get_size_for_crop(image: cv2.Mat) -> int:
    height = image.shape[0]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # intesity of pixels on the y axis
    intensity_y, y = [], []
    for i in range(height):
        y.append(i)
        intensity_y.append(image_gray[i, 0])
    
    # plt.plot(y, intensity_y)
    # plt.show()

    # find y-coord where black description bar starts
    for i in range(height-2, -1, -1):
        if intensity_y[i] - intensity_y[i+1] > 0:
            start_description = i+1
            break
    
    return height - start_description

def filter_image(image: cv2.Mat, filter: str, average_ksize=5, median_ksize=7,
                gauss_ksize=5, x_deviation=1, y_deviation=1) -> cv2.Mat:

    if filter == 'Average':
        filtered_image = cv2.blur(image, (average_ksize, average_ksize))

    elif filter == 'Median':
        if median_ksize % 2 != 1:
            median_ksize = 7
        filtered_image = cv2.medianBlur(image, median_ksize)

    elif filter == 'Gaussian':
        if gauss_ksize % 2 != 1:
            gauss_ksize = 5
        filtered_image = cv2.GaussianBlur(image, (gauss_ksize, gauss_ksize), x_deviation, y_deviation)

    else:
        filtered_image = image

    return filtered_image

def plot_intensity_dist(image: cv2.Mat, bins: int, figsize: tuple):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=figsize)
    plt.title('Intensity histogram')
    ax = plt.hist(image_gray.ravel(), bins, [0, 256])
    # ax = cv2.calcHist([image_gray], [0], None, [bins], [0, 256])
    # plt.plot(ax)
    plt.show()

    return ax

def morph_transform(image: cv2.Mat, operation: str, ksize=5, iters=1):

    kernel = np.ones((ksize, ksize), np.uint8)

    if operation == 'erosion':
        transformed_image = cv2.erode(image, kernel, iterations=iters)
    
    elif operation == 'dilation':
        transformed_image = cv2.dilate(image, kernel, iterations=iters)
    
    elif operation == 'opening':
        transformed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iters)

    elif operation == 'closing':
        transformed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iters)

    else:
        transformed_image = image

    return transformed_image

def detect_edges(image: cv2.Mat, algorithm: str, gauss_ksize=3, sobel_ksize=5,
                dx=1, dy=1, treshold1=100, treshold2=200):
    
    if algorithm == 'Sobel':
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if sobel_ksize not in (1, 3, 5, 7):
            sobel_ksize = 5

        blur_params = {'gauss_ksize': gauss_ksize, 'x_deviation': 0, 'y_deviation': 0}
        image_blur = filter_image(image_gray, 'Gaussian', **blur_params)
        edged_image = cv2.Sobel(image_blur, cv2.CV_64F, dx=dx, dy=dy, ksize=sobel_ksize)
    
    elif algorithm == 'Canny':
        edged_image = cv2.Canny(image, treshold1, treshold2)

    else:
        edged_image = image
    
    return edged_image