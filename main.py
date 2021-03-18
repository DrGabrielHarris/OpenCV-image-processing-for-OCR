# %%
from pathlib import Path

import cv2

from process import (
    apply_adaptive_thresholding,
    apply_gaussian_smoothing,
    apply_laplacian_filter,
    apply_morphological_operation,
    apply_sobel_filter,
)
from visualize import add_white_boarder, crop_image, resize_image, show_image

# %%
# read image, convert to RGB, and show
img = cv2.imread(
    filename=Path(
        "images",
        "hands-on-machine-learning.jpg",
    ),
    flags=cv2.IMREAD_COLOR,
)

h, w, c = img.shape
print(f"Image shape: {h}H x {w}W x {c}C")

img.dtype

img = cv2.cvtColor(
    src=img,
    code=cv2.COLOR_BGR2RGB,
)

show_image(img)

# %%
# crop image and show
img = crop_image(img, ymin=200, ymax=780, xmin=100, xmax=1000)

h, w, c = img.shape
print(f"Image shape: {h}H x {w}W x {c}C")

show_image(img)

# %%
# add 10 pixels wide border to cropped image
img_border = add_white_boarder(img, 10)

h, w, c = img_border.shape
print(f"Image shape: {h}H x {w}W x {c}C")

show_image(img_border)

# %%
# resize image
MAX_PIX = 800

if h > MAX_PIX:
    img_resized = resize_image(img, "h", MAX_PIX)
    show_image(img_resized)

if w > MAX_PIX:
    img_resized = resize_image(img, "w", MAX_PIX)
    show_image(img_resized)

# %%
# apply morphology

img_opened = apply_morphological_operation(img, "open")
show_image(img_opened)

img_close = apply_morphological_operation(img, "close")
show_image(img_close)

# %%
# Apply Gaussian smoothing

img_gaussian = apply_gaussian_smoothing(img)
show_image(img_gaussian)

# %%
#  Apply adaptive thresholding

img_adaptive_gaussian = apply_adaptive_thresholding(img, "gaussian")
show_image(img_adaptive_gaussian, cmap="gray")

img_adaptive_mean = apply_adaptive_thresholding(img, "mean")
show_image(img_adaptive_mean, cmap="gray")


# %%
# Apply Sobel filter

img_sobel_composit = apply_sobel_filter(img, "h") + apply_sobel_filter(img, "v")
show_image(img_sobel_composit, cmap="gray")


# %%
# Apply Laplacian filter

img_laplacian = apply_laplacian_filter(img)
show_image(img_laplacian, cmap="gray")

# %%
_, buf = cv2.imencode(
    ext=".PNG",
    img=img,
)

data = buf.tostring()

img = cv2.imdecode(
    buf=buf,
    flags=cv2.IMREAD_UNCHANGED,
)

# %%
