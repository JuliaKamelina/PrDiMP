import sys
import os
import cv2
import numpy as np
import argparse

from PIL import Image
sys.path.append('./')
sys.path.append('./implementation/pytracking')

from implementation import PrDiMPMUTracker
from implementation.utils import load_video_info, get_sequence_info

def demo_tracker(video_path, net_path, no_show):
    seq, ground_truth = load_video_info(video_path)
    seq = get_sequence_info(seq)
    frames = [np.array(Image.open(f)) for f in seq["image_files"]]
    is_color = True if (len(frames[0].shape) == 3) else False
    tracker = PrDiMPMUTracker(frames[0].shape[:2], net_path, is_color, mu_model_dir='prdimp_mu_1/')
    for i, frame in enumerate(frames):
        if i == 0:
            output = tracker.initializing(frame, seq, reuse=False)
        else:
            output, _ = tracker.tracking(frame)
        bbox = output.get('target_bbox', seq["init_rect"])
        bbox = (bbox[0], bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3])
        time = output.get('time', 0)
        print(i)
        print('bb: ', bbox)
        print(time)
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 255), 1)
        gt_bbox = (ground_truth[i, 0],
                   ground_truth[i, 1],
                   ground_truth[i, 0] + ground_truth[i, 2],
                   ground_truth[i, 1] + ground_truth[i, 3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]), int(gt_bbox[1])),
                              (int(gt_bbox[2]), int(gt_bbox[3])),
                              (0, 255, 0), 1)
        print('gt: ', gt_bbox)
        print("#######################################################################")
        if not no_show:
            cv2.imshow('', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="sequences/Couple")
    parser.add_argument("--net_path", default="networks/prdimp50.pth.tar")
    parser.add_argument("--no_show", action='store_true')
    args = parser.parse_args()
    demo_tracker(args.video, args.net_path, args.no_show)
