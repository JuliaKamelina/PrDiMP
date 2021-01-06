import numpy as np
import math
import torch
import torch.nn.functional as F

from ..runfiles import settings
from ..fourier_tools import max2d
from ..utils import TensorList

from ..pytracking.ltr.models.layers import activation
from ..pytracking.ltr.data.bounding_box_utils import  rect_to_rel, rel_to_rect

def localize_target(scores, sample_pos, sample_scales, tracker):
    scores = scores.squeeze(1)

    preprocess_method = settings.score_preprocess
    if preprocess_method == 'none':
        pass
    elif preprocess_method == 'exp':
        scores = scores.exp()
    elif preprocess_method == 'softmax':
        reg_val = getattr(tracker.features.net.classifier.filter_optimizer, 'softmax_reg', None)
        scores_view = scores.view(scores.shape[0], -1)
        scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
        scores = scores_softmax.view(scores.shape)
    else:
        raise Exception('Unknown score_preprocess in params.')

    score_filter_ksz = getattr(settings, 'score_filter_ksz', 1)
    if score_filter_ksz > 1:
        assert score_filter_ksz % 2 == 1
        kernel = scores.new_ones(1, 1, score_filter_ksz, score_filter_ksz)
        scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

    if getattr(settings, 'advanced_localization', False):
        return localize_advanced(scores, sample_pos, sample_scales, tracker)

    # Get maximum
    score_sz = torch.Tensor(list(scores.shape[-2:]))
    score_center = (score_sz - 1)/2
    max_score, max_disp = max2d(scores)
    _, scale_ind = torch.max(max_score, dim=0)
    max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
    target_disp = max_disp - score_center

    # Compute translation vector and scale change factor
    output_sz = score_sz - (tracker.kernel_size + 1) % 2
    translation_vec = target_disp * (tracker.img_support_sz / output_sz) * sample_scales[scale_ind]

    return translation_vec, scale_ind, scores, None

def localize_advanced(scores, sample_pos, sample_scales, tracker):
    sz = scores.shape[-2:]
    score_sz = torch.Tensor(list(sz))
    output_sz = score_sz - (tracker.kernel_size + 1) % 2
    score_center = (score_sz - 1)/2

    scores_hn = scores
    if tracker.output_window is not None and getattr(settings, 'perform_hn_without_windowing', False):
        scores_hn = scores.clone()
        scores *= tracker.output_window

    max_score1, max_disp1 = max2d(scores)
    _, scale_ind = torch.max(max_score1, dim=0)
    sample_scale = sample_scales[scale_ind]
    max_score1 = max_score1[scale_ind]
    max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
    target_disp1 = max_disp1 - score_center
    translation_vec1 = target_disp1 * (tracker.img_support_sz / output_sz) * sample_scale

    if max_score1.item() < settings.target_not_found_threshold:
        return translation_vec1, scale_ind, scores_hn, 'not_found'
    if max_score1.item() < getattr(settings, 'uncertain_threshold', -float('inf')):
        return translation_vec1, scale_ind, scores_hn, 'uncertain'
    if max_score1.item() < getattr(settings, 'hard_sample_threshold', -float('inf')):
        return translation_vec1, scale_ind, scores_hn, 'hard_negative'

    # Mask out target neighborhood
    target_neigh_sz = settings.target_neighborhood_scale * (tracker.target_sz / sample_scale) * (output_sz / tracker.img_support_sz)

    tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
    tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
    tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
    tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
    scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
    scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

    # Find new maximum
    max_score2, max_disp2 = max2d(scores_masked)
    max_disp2 = max_disp2.float().cpu().view(-1)
    target_disp2 = max_disp2 - score_center
    translation_vec2 = target_disp2 * (tracker.img_support_sz / output_sz) * sample_scale

    prev_target_vec = (tracker.pos - sample_pos[scale_ind,:]) / ((tracker.img_support_sz / output_sz) * sample_scale)

    # Handle the different cases
    if max_score2 > settings.distractor_threshold * max_score1:
        disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
        disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
        disp_threshold = settings.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

        if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'
        if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
            return translation_vec2, scale_ind, scores_hn, 'hard_negative'
        if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        # If also the distractor is close, return with highest score
        return translation_vec1, scale_ind, scores_hn, 'uncertain'

    if max_score2 > settings.hard_negative_threshold * max_score1 and max_score2 > settings.target_not_found_threshold:
        return translation_vec1, scale_ind, scores_hn, 'hard_negative'

    return translation_vec1, scale_ind, scores_hn, 'normal'

def refine_target_box(tracker, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
    if hasattr(tracker.features.net.bb_regressor, 'predict_bb'):
        return direct_box_regression(tracker, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

    # Initial box for refinement
    init_box = tracker.get_iounet_box(tracker.pos, tracker.target_sz, sample_pos, sample_scale)

    # Extract features from the relevant scale
    iou_features = tracker.get_iou_features(backbone_feat)
    iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

    # Generate random initial boxes
    init_boxes = init_box.view(1,4).clone()
    if settings.num_init_random_boxes > 0:
        square_box_sz = init_box[2:].prod().sqrt()
        rand_factor = square_box_sz * torch.cat([settings.box_jitter_pos * torch.ones(2), settings.box_jitter_sz * torch.ones(2)])

        minimal_edge_size = init_box[2:].min()/3
        rand_bb = (torch.rand(settings.num_init_random_boxes, 4) - 0.5) * rand_factor
        new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
        new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
        init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
        init_boxes = torch.cat([init_box.view(1,4), init_boxes])

    # Optimize the boxes
    output_boxes, output_iou = optimize_boxes_relative(tracker, iou_features, init_boxes)

    # Remove weird boxes
    output_boxes[:, 2:].clamp_(1)
    aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
    keep_ind = (aspect_ratio < settings.maximal_aspect_ratio) * (aspect_ratio > 1/settings.maximal_aspect_ratio)
    output_boxes = output_boxes[keep_ind,:]
    output_iou = output_iou[keep_ind]

    # If no box found
    if output_boxes.shape[0] == 0:
        return

    # Predict box
    k = settings.iounet_k
    topk = min(k, output_boxes.shape[0])
    _, inds = torch.topk(output_iou, topk)
    predicted_box = output_boxes[inds, :].mean(0)
    predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

    # Get new position and size
    new_pos = predicted_box[:2] + predicted_box[2:] / 2
    new_pos = (new_pos.flip((0,)) - (tracker.img_sample_sz - 1) / 2) * sample_scale + sample_pos
    new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
    new_scale = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())

    tracker.pos_iounet = new_pos.clone()

    # if self.params.get('use_iounet_pos_for_learning', True):
    tracker.pos = new_pos.clone()

    tracker.target_sz = new_target_sz

    if update_scale:
        tracker.target_scale = new_scale

def direct_box_regression(tracker, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
    """Implementation of direct bounding box regression."""

    # Initial box for refinement
    init_box = tracker.get_iounet_box(tracker.pos, tracker.target_sz, sample_pos, sample_scale)

    # Extract features from the relevant scale
    iou_features = tracker.get_iou_features(backbone_feat)
    iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

    # Generate random initial boxes
    init_boxes = init_box.view(1, 1, 4).clone().to(settings.device)

    # Optimize the boxes
    output_boxes = tracker.features.net.bb_regressor.predict_bb(tracker.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

    # Remove weird boxes
    output_boxes[:, 2:].clamp_(1)

    predicted_box = output_boxes[0, :]

    # Get new position and size
    new_pos = predicted_box[:2] + predicted_box[2:] / 2
    new_pos = (new_pos.flip((0,)) - (tracker.img_sample_sz - 1) / 2) * sample_scale + sample_pos
    new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
    new_scale_bbr = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())
    new_scale = new_scale_bbr

    tracker.pos_iounet = new_pos.clone()

    tracker.pos = new_pos.clone()

    tracker.target_sz = new_target_sz

    if update_scale:
        tracker.target_scale = new_scale

def optimize_boxes_relative(tracker, iou_features, init_boxes):
    """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
    output_boxes = init_boxes.view(1, -1, 4).to(settings.device)
    step_length = settings.box_refinement_step_length
    if isinstance(step_length, (tuple, list)):
        step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(settings.device).view(1,1,4)

    sz_norm = output_boxes[:,:1,2:].clone()
    output_boxes_rel = rect_to_rel(output_boxes, sz_norm)
    for _ in range(settings.box_refinement_iter):
        # forward pass
        bb_init_rel = output_boxes_rel.clone().detach()
        bb_init_rel.requires_grad = True

        bb_init = rel_to_rect(bb_init_rel, sz_norm)
        outputs = tracker.features.net.bb_regressor.predict_iou(tracker.iou_modulation, iou_features, bb_init)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        outputs.backward(gradient = torch.ones_like(outputs))

        # Update proposal
        output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
        output_boxes_rel.detach_()

        step_length *= settings.box_refinement_step_decay

    #     for s in outputs.view(-1):
    #         print('{:.2f}  '.format(s.item()), end='')
    #     print('')
    # print('')

    output_boxes = rel_to_rect(output_boxes_rel, sz_norm)

    return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()
