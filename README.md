# Processing biological images using Python and OpenCV

## Task 1
Create a module utils.py for more comfortable working with images.

Neccessary functions:

- delete_description(image, size)

- plot_density_dist(image, bins, figsize)


- filter_image(image, filter, params)
https://plainenglish.io/blog/image-filtering-and-editing-in-python-with-code-e878d2a4415d

- morph_transform(image, operation, params)
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

- detect_edges(image, algorithm, params)
https://learnopencv.com/edge-detection-using-opencv/
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

Create .ipynb to show how it's working.


## Task 2
Create a file cell_detection.py which contains function get_cells(image). This function is a wrapper for algorithms giving mask fo source image. If the picture area is a cell then it has color white, otherwise it has color black.

Realize algorithm using SimpleBlobDetector:
https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/
https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html

On the preprocessing stage apply blur filter. Use morphological operations (opening / closing) to get purer masks. All of this functions are implemented in utils.py.

Create .ipynb to show how it's working.