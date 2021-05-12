import numpy as np
from PIL import Image
from scipy.misc import imresize


def crop_image(img, bbox, img_size=107, padding=16, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape

    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        if min_x_val == max_x_val:
            if min_x_val <= 0:
                min_x_val = 0
                max_x_val = min_x_val + 3
            if max_x_val >= img_w - 1:
                max_x_val = img_w - 1
                min_x_val = img_w - 4
        if min_y_val == max_y_val:
            if min_y_val <= 0:
                min_y_val = 0
                max_y_val = 3
            if max_y_val >= img_h - 1:
                max_y_val = img_h - 1
                min_y_val = img_h - 4

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    try:
        scaled = imresize(cropped, (img_size, img_size))
    except ValueError:
        print("a")
    return scaled

def me_extract_regions(image, samples, crop_size=107, padding=16, shuffle=False):
    regions = np.zeros((samples.shape[0], crop_size, crop_size, 3), dtype='uint8')
    for t in range(samples.shape[0]):
        regions[t] = crop_image(image, samples[t], crop_size, padding)

    regions = regions #- 128.
    return regions
