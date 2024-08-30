# meta class, backbone
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """All backbones should have a init to specify their own set of parameters"""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ All backbones should have a forward method
        
        Must return a dict of features with keys:
        """
        pass

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )