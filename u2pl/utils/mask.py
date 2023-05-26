"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np


import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math

class GridMaskGenerator(object):
    def __init__(self, patch_size):
        """mode: 0/1 for masked reigons"""
        self.d = patch_size

    def __single_mask__(self, h, w):        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        d = self.d * 2
        
        self.l = self.d
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(1)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        mask = torch.from_numpy(mask).float()

        return mask

    def __call__(self, x):
        n,c,h,w = x.size()
        y = []
        for i in range(n):
            y.append(self.__single_mask__(h, w))
        y = torch.cat(y).view(n,1,h,w).to(device=x.device)
        return y


class RandomMaskGenerator(object):
    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape
        if isinstance(H, int):
            mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        else:
            mshape = B, 1, round(H.item() / self.mask_block_size), round(
                W.item() / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = F.interpolate(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def __call__(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask

class BlockMaskGenerator(object):
    def __init__(self, mask_ratio, min_num_patches=4, max_num_patches=None, min_aspect=0.3, max_aspect=None):
        self.mask_ratio = mask_ratio

        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))


    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta
    
    def __single_mask__(self, h, w):
        self.height = h
        self.width = w
        num_masking_patches = h * w * self.mask_ratio
        max_num_patches = num_masking_patches if self.max_num_patches is None else self.max_num_patches
        mask = np.zeros(shape=(h, w), dtype=np.int)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

    def __call__(self, x):
        n,c,h,w = x.size()
        y = []
        for i in range(n):
            y.append(torch.from_numpy(self.__single_mask__(h, w)))
        y = torch.cat(y).view(n,1,h,w).to(device=x.device)
        return 1 - y