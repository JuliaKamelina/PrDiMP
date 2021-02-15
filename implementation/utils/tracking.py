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