import os
import importlib
import inspect

import numpy as np
import torch
import torch.nn.functional as F

from ..runfiles import settings

class PrDiMPFeatures:
    def __init__(self, is_color, net_path, device='cpu'):
        self.is_color = is_color
        self.net_path = net_path
        _mean = (0.485, 0.456, 0.406)
        _std = (0.229, 0.224, 0.225)
        self._mean = torch.Tensor(_mean).view(1, -1, 1, 1)
        self._std = torch.Tensor(_std).view(1, -1, 1, 1)
        self.device = device
        self.load_network()

    def load_network(self, backbone_pretrained=False):
        weight_dict = torch.load(self.net_path, map_location='cpu')
        net_constr = weight_dict['constructor']
        net_fun = getattr(importlib.import_module(net_constr.fun_module), net_constr.fun_name)
        net_fun_args = list(inspect.signature(net_fun).parameters.keys())
        if 'backbone_pretrained' in net_fun_args:
            net_constr.kwds['backbone_pretrained'] = backbone_pretrained
        self.net = net_constr.get()
        self.net.load_state_dict(weight_dict['net'])
        self.net.constructor = weight_dict['constructor']
        if self.device == 'cuda':
            self.net.cuda()
        self.net.eval()

    def sample_patch_transformed(self, im, pos, scale, image_sz, transforms):
        im_patch, _ = self.get_sample(im, pos, scale*image_sz, image_sz)
        im_patches = torch.cat([T(im_patch, is_mask=False) for T in transforms])
        return im_patches

    def sample_patch_multiscale(self, im, pos, scales, image_sz, mode: str='replicate', max_scale_change=None):
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        patch_iter, coord_iter = zip(*(self.get_sample(im, pos, s*image_sz, image_sz, mode=mode,
                                                    max_scale_change=max_scale_change) for s in scales))
        im_patches = torch.cat(list(patch_iter))
        patch_coords = torch.cat(list(coord_iter))

        return  im_patches, patch_coords

    def get_sample(self, im, pos, img_sample_sz, output_sz, mode='replicate', max_scale_change=None):
        posl = pos.clone().long()

        pad_mode = mode
        if mode == 'inside' or mode == 'inside_major':
            pad_mode = 'replicate'
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            shrink_factor = (img_sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=max_scale_change)
            img_sample_sz = (img_sample_sz.float() / shrink_factor).long()

        if output_sz is not None:
            resize_factor = torch.min(img_sample_sz.float() / output_sz.float()).item()
            df = int(max(int(resize_factor - 0.1), 1))
        else:
            df = int(1)
        sz = img_sample_sz.float() / df

        if df > 1:
            os = posl % df              # offset
            posl = (posl - os) // df     # new position
            im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
        else:
            im2 = im

        # sz = torch.Tensor(sz)
        szl = torch.max(sz.round(), torch.Tensor([2])).long()

        # Extract top and bottom coordinates
        tl = posl - (szl - 1) // 2
        br = posl + szl//2 + 1

        if mode == 'inside' or mode == 'inside_major':
            im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
            shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
            tl += shift
            br += shift

            outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
            shift = (-tl - outside) * (outside > 0).long()
            tl += shift
            br += shift

        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
        patch_coord = df * torch.cat((tl, br)).view(1, 4)

        if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
            return im_patch.clone(), patch_coord

        # output_sz = torch.Tensor(output_sz)
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')

        return im_patch, patch_coord

    def preprocess_image(self, im):
        im = im/255
        im -= self._mean
        im /= self._std

        if self.device == 'cuda':
            im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        im = self.preprocess_image(im)
        return self.net.extract_backbone_features(im)
