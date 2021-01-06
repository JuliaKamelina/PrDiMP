import numpy as np
import cv2 as cv

def get_sequence_info(seq):
    if not 'format' in seq.keys():
        seq['format'] = 'vot'  # TODO: CHECK!!!

    seq['frame'] = 0
    if (seq['format'] == 'otb'):  # TODO: if seq['frame'] == 'vot'
        seq['init_sz'] = np.array([seq['init_rect'][3], seq['init_rect'][2]])
        seq['init_pos'] = np.array([seq['init_rect'][1], seq['init_rect'][0]]) + (seq["init_sz"] - 1)/2
        seq['num_frames'] = len(seq['image_files'])
        seq['rect_position'] = np.zeros((seq['num_frames'], 4))

    return seq
