import math
import numpy as np
import time
import torch
import torch.nn.functional as F

from .feature_extraction import PrDiMPFeatures, augmentation
from .localization import localize_target, refine_target_box
from .runfiles import settings
from .utils import TensorList, plot_graph


class PrDiMPTracker:
    def __init__(self, image_sz, net_path, is_color=True):
        self.is_color_image = is_color
        # self.pos = torch.Tensor(seq["init_pos"])
        # self.target_sz = torch.Tensor(seq["init_sz"])
        self.frame_num = 1
        self.features = PrDiMPFeatures(is_color, net_path, settings.device)
        # im = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0)
        self.image_sz = torch.Tensor(image_sz)
        sz = [settings.image_sample_size, settings.image_sample_size]
        self.img_sample_sz = torch.Tensor(sz)
        self.img_support_sz = self.img_sample_sz
        # search_area = torch.prod(self.target_sz * settings.search_area_scale).item()
        # self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()
        # self.base_target_sz = self.target_sz / self.target_scale
        # if not hasattr(settings, 'scale_factors'):
        #     settings.scale_factors = torch.ones(1)
        # self.min_scale_factor = torch.max(10 / self.base_target_sz)
        # self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)
    
    def initialize(self, im, seq):
        self.pos = torch.Tensor(seq["init_pos"])
        self.target_sz = torch.Tensor(seq["init_sz"])
        search_area = torch.prod(self.target_sz * settings.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()
        self.base_target_sz = self.target_sz / self.target_scale
        if not hasattr(settings, 'scale_factors'):
            settings.scale_factors = torch.ones(1)
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        tic = time.time()

        im = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0)
        init_backbone_feat = self.generate_init_samples(im)

        self.init_classifier(init_backbone_feat)
        self.init_iou_net(init_backbone_feat)

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict=None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = augmentation.numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * settings.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        with torch.no_grad():
            scores_raw = self.features.net.classifier.classify(self.target_filter, test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = localize_target(scores_raw, sample_pos, sample_scales, self)
        new_pos = sample_pos[scale_ind,:] + translation_vec

        # Update position and scale
        if flag != 'not_found':
            # if self.params.get('use_iou_net', True):
            update_scale_flag = settings.update_scale_when_uncertain or flag != 'uncertain'
            # if self.params.get('use_classifier', True):
            self.update_state(new_pos)
            refine_target_box(self, backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            # elif self.params.get('use_classifier', True):
            #     self.update_state(new_pos, sample_scales[scale_ind])


        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = settings.hard_negative_learning_rate if hard_negative else None

        if update_flag and settings.update_classifier:
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])

        # Set the pos of the tracker to iounet pos
        if getattr(settings, 'use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.debug_info['flag'] = flag
        self.debug_info['max_score'] = max_score

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]] - 1)/2, self.target_sz[[1,0]]))

        if getattr(settings, 'output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state}
        return out, [new_state.tolist(), score_map.cpu().data.numpy(), test_x, scale_ind, sample_pos, sample_scales, flag, s]

    def get_centered_sample_pos(self):
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = self.features.sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=settings.border_mode,
                                                           max_scale_change=settings.patch_max_scale_change)
        with torch.no_grad():
            im_patches = self.features.preprocess_image(im_patches)
            backbone_feat = self.features.net.extract_backbone_features(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_sample_location(self, sample_coord):
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = getattr(settings, 'target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = settings.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % getattr(settings, 'train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        # low_score_th = None
        if hard_negative_flag:
            num_iter = settings.net_opt_hn_iter
        # elif low_score_th is not None and low_score_th > scores.max().item():
        #     num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % settings.train_skipping == 0:
            num_iter = settings.net_opt_update_iter

        plot_loss = settings.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.features.net.classifier.filter_optimizer(self.target_filter,
                                                                                              num_iter=num_iter, feat=samples,
                                                                                              bb=target_boxes,
                                                                                              sample_weight=sample_weights,
                                                                                              compute_losses=plot_loss)

            if settings.debug:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))

    def generate_init_samples(self, im):
        self.init_sample_scale = self.target_scale
        global_shift = torch.zeros(2)
        self.init_sample_pos = self.pos.round()
        aug_expansion_factor = settings.augmentation_expansion_factor
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        get_rand_shift = lambda: None
        random_shift_factor = getattr(settings, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = settings.augmentation if settings.use_augmentation else {}
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        im_patches = self.features.sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)
        with torch.no_grad():
            im_patches = self.features.preprocess_image(im_patches)
            init_backbone_feat = self.features.net.extract_backbone_features(im_patches)

        return init_backbone_feat

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.features.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.features.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.features.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.features.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        # self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in settings.augmentation and getattr(settings, 'use_augmentation', True):
            num, prob = settings.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.features.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        # if self.params.get('window_output', False):
        #     if self.params.get('use_clipped_window', False):
        #         self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
        #     else:
        #         self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
        #     self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = settings.debug
        num_iter = settings.net_opt_iter

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.features.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                                    compute_losses=plot_loss)

        # Init memory
        if settings.update_classifier:
            self.init_memory(TensorList([x]))

        if settings.debug:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if settings.debug:
                plot_graph(self.losses, 10, title='Training Loss')

    def init_target_boxes(self):
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(settings.device)
        self.target_boxes = init_target_boxes.new_zeros(settings.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes
    
    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        # box_sz = torch.Tensor(box_sz)
        # target_ul = torch.Tensor(target_ul)
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(settings.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(settings.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = settings.learning_rate

            init_samp_weight = getattr(settings, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.features.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(settings.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])
