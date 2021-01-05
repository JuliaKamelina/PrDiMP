import numpy as np

class PrDiMPParams:
    device = 'cuda'
    debug = True
    image_sample_size = 22*16
    search_area_scale = 6
    border_mode = 'inside_major'
    patch_max_scale_change = 1.5

    # Learning parameters
    sample_memory_size = 50
    learning_rate = 0.01
    init_samples_minimum_weight = 0.25
    train_skipping = 20

    # Net optimization params
    update_classifier = True
    net_opt_iter = 10
    net_opt_update_iter = 2
    net_opt_hn_iter = 1

    # Detection parameters
    window_output = False

    # Init augmentation parameters
    use_augmentation = True
    augmentation = {'fliplr': True,
                    'rotate': [10, -10, 45, -45],
                    'blur': [(3,1), (1, 3), (2, 2)],
                    'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                    'dropout': (2, 0.2)}

    augmentation_expansion_factor = 2
    random_shift_factor = 1/3

    # Advanced localization parameters
    advanced_localization = True
    score_preprocess = 'softmax'
    target_not_found_threshold = 0.04
    distractor_threshold = 0.8
    hard_negative_threshold = 0.5
    target_neighborhood_scale = 2.2
    dispalcement_scale = 0.8
    hard_negative_learning_rate = 0.02
    update_scale_when_uncertain = True

    # IoUnet parameters
    box_refinement_space = 'relative'
    iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    iounet_k = 3                     # Top-k average to estimate final box
    num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    maximal_aspect_ratio = 6         # Limit on the aspect ratio
    box_refinement_iter = 10          # Number of iterations for refining the boxes
    box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    box_refinement_step_decay = 1 
