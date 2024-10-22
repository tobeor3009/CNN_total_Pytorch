""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc
from torch import nn


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

