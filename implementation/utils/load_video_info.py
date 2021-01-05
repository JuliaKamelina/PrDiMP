import numpy as np
import os.path

def load_video_info(path):
    gt_path = path #"/".join(path.split("/")[:-2])
    gt_path = "{}/groundtruth_rect.txt".format(gt_path)
    try:
        ground_truth = np.loadtxt(gt_path)
    except:
        ground_truth = np.loadtxt(gt_path, delimiter=",")

    seq = dict()
    seq["format"] = "otb"
    seq["len"] = len(ground_truth)
    seq["init_rect"] = ground_truth[0]

    img_path = path + '/img/'
    img_files = list()
    if (os.path.isfile(img_path + '%04d' % 1 + '.png')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.png')
    elif (os.path.isfile(img_path + '%04d' % 1 + '.jpg')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')
    elif (os.path.isfile(img_path + '%05d' % 1 + '.jpg')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%05d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 3 + '.jpg')):
        for i in range(3,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 247 + '.jpg')):
        for i in range(247,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 18 + '.jpg')):
        for i in range(18,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 1 + '.bmp')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.bmp')
    else:
         raise ValueError("No images loaded")

    seq["image_files"] = img_files
    return(seq, ground_truth)

def load_video_info_test(path):
    gt_path = "/".join(path.split("/")[:-2])
    gt_path = "{}/groundtruth_rect.txt".format(gt_path)
    try:
        ground_truth = np.loadtxt(gt_path)
    except:
        ground_truth = np.loadtxt(gt_path, delimiter=",")

    seq = dict()
    seq["format"] = "otb"
    seq["len"] = len(ground_truth)
    seq["init_rect"] = ground_truth[0]

    img_path = path
    img_files = list()
    if (os.path.isfile(img_path + '%04d' % 1 + '.png')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.png')
    elif (os.path.isfile(img_path + '%04d' % 1 + '.jpg')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')
    elif (os.path.isfile(img_path + '%05d' % 1 + '.jpg')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%05d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 3 + '.jpg')):
        for i in range(3,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 247 + '.jpg')):
        for i in range(247,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 18 + '.jpg')):
        for i in range(18,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.jpg')

    elif (os.path.isfile(img_path + '%04d' % 1 + '.bmp')):
        for i in range(1,seq["len"]):
            img_files.append(img_path + '%04d' % i + '.bmp')
    else:
         raise ValueError("No images loaded")

    seq["image_files"] = img_files
    return(seq, ground_truth)
