import cv2
import matplotlib.pyplot as plt


def show_image(img, **kwargs):
    """Show an image without any interpolation

    Args:
        img: image as numpy array
    """
    plt.subplot()
    plt.axis("off")
    plt.imshow(X=img, interpolation="none", **kwargs)


def crop_image(img, ymin, ymax, xmin, xmax):
    """Crop image with given size

    Args:
        img: image as numpy array
        ymin: start cropping position along height in pixels
        ymax: end cropping position along height in pixels
        xmin: end cropping position along width in pixels
        xmax: end cropping position along width in pixels

    Returns:
        Image as numpy array
    """
    return img[int(ymin) : int(ymax), int(xmin) : int(xmax), :]


def add_white_boarder(img, width):
    """Add a white boarder to all sides of an image

    Args:
        img: image as numpy array
        width: boarder width in pixels

    Returns:
        Image as numpy array
    """
    return cv2.copyMakeBorder(
        src=img,
        top=width,
        bottom=width,
        left=width,
        right=width,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def resize_image(img, direction, MAX_PIX):
    """
    Resize an image along the height or the width, and keep its aspect ratio

    Args:
        img: image as numpy array
        direction: either 'h' or 'v'
        MAX_PIX: required maximum number of pixels

    Returns:
        Image as numpy array
    """
    h, w, c = img.shape

    if direction == "h":
        dsize = (int((MAX_PIX * w) / h), int(MAX_PIX))

    elif direction == "v":
        dsize = (int(MAX_PIX), int((MAX_PIX * h) / w))

    img_resized = cv2.resize(
        src=img,
        dsize=dsize,
        interpolation=cv2.INTER_CUBIC,
    )

    h, w, c = img_resized.shape
    print(f"Image shape: {h}H x {w}W x {c}C")

    return img_resized
