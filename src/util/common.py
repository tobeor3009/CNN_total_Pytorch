""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc
from torch import nn
import cv2
import numpy as np

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def set_dropout_probability(model, decay_dropout_ratio=0.95):
    for _, module in model.named_modules():
        # 모듈이 Dropout 또는 Dropout2d, Dropout3d라면
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            previous_p = module.p
            new_p = previous_p * decay_dropout_ratio
            module.p = new_p

def create_overlay_image(image_array, mask_array, alpha=1.0, color_tuple=(255, 0, 0)):
    # 원본 이미지를 3채널로 변환
    image_normalized = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_array_color = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
    # 마스크가 적용된 영역에만 빨간색 추가
    
    mask_array = mask_array[..., None].repeat(3, -1)
    color_mask_array = cv2.normalize(mask_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_mask_array = (color_mask_array > 127) * np.array(color_tuple)[None, None]
    color_mask_array = color_mask_array.astype("uint8")
    # 마스크가 있는 부분에만 빨간색을 합성
    overlayed_image = image_array_color.copy()
    overlayed_image = cv2.addWeighted(overlayed_image, 1.0, color_mask_array, alpha, 0.0)
    
    return overlayed_image