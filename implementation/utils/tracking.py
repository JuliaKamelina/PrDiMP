import numpy as np

def process_regions(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions / 255.0
    regions[:, :, :, 0] = (regions[:, :, :, 0] - 0.485) / 0.229
    regions[:, :, :, 1] = (regions[:, :, :, 1] - 0.456) / 0.224
    regions[:, :, :, 2] = (regions[:, :, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (0, 3, 1, 2))
    # regions = np.expand_dims(regions, axis=0)
    # regions = np.tile(regions, (2,1,1,1))

    return regions

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou
