# %%
import cv2
import numpy as np


def apply_morphological_operation(img, method):
    """Apply a morphological operation depending on method

    Opening: erosion followed by dilation and is useful for removing noise
    Closing: dilation followed by erosion and is useful for closing small holes

    Args:
        img: image as numpy array
        method: either 'open' for opening or 'close' for closing

    Returns:
        Image as numpy array

    """
    if method == "open":
        op = cv2.MORPH_OPEN
    elif method == "close":
        op = cv2.MORPH_CLOSE

    return cv2.morphologyEx(
        src=img,
        op=op,
        kernel=np.ones((5, 5), np.uint8),
    )


def apply_gaussian_smoothing(img):
    """Apply Gaussian smoothing with 5x5 kernal size

    Useful for removing high frequency content (e.g. noise)

    Args:
        img: image as numpy array

    Returns:
        Image as numpy array
    """
    return cv2.GaussianBlur(
        src=img,
        ksize=(5, 5),
        sigmaX=0,
        sigmaY=0,
    )


def apply_adaptive_thresholding(img, method):
    """Apply adaptive thresholding with a threshold value depending on method

    Threshold value is:
    Gaussian:  sum of neighborhood values where weights are a Gaussian window
    Mean: mean of neighborhood area

    Useful when the image has different lighting conditions in different areas

    Args:
        img: image as numpy array
        method: either 'gaussian' for 'mean'

    Returns:
        Image as numpy array
    """
    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    if method == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    elif method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

    return cv2.adaptiveThreshold(
        src=img,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )


def apply_sobel_filter(img, direction):
    """
    Apply Sobel filter of first order (i.e. 1st derivative) along direction

    Direction could be along x (horizontally) or y (vertically)


    Useful to detect horizontal or vertical edges and are resistant to noise

    Args:
        img: image as numpy array
        direction: either 'h' or 'v'

    Returns:
        Image as numpy array
    """
    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    if direction == "h":
        dx, dy = 0, 1

    elif direction == "v":
        dx, dy = 1, 0

    return cv2.Sobel(
        src=img,
        ddepth=cv2.CV_64F,
        dx=dx,
        dy=dy,
        ksize=3,
    )


def apply_laplacian_filter(img):
    """
    Apply Laplacian filter of second order (i.e. 2nd derivative) along both x (horizontally) and y (vertically)

    Useful to detect edges

    Args:
        img: image as numpy array

    Returns:
        Image as numpy array
    """
    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    return np.uint8(
        np.absolute(
            cv2.Laplacian(
                src=img,
                ddepth=cv2.CV_64F,
            )
        )
    )
