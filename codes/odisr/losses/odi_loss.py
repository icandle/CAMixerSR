import torch
from torch import nn as nn
from torch.nn import functional as F
import math
import numpy as np

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss
from basicsr.losses.basic_loss import l1_loss, L1Loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class WSL1Loss(nn.Module):
    """WS-L1 loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, h=1024, reduction='mean'):
        super(WSL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        # _weight = [math.cos((i - (h / 2) + 0.5) * math.pi / h) for i in range(h)]
        self.h = h

    def forward(self, pred, target, top, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
            top: (Tensor): of shape (N, H). Start latitude of patches.
        """
        _weight = top.unsqueeze(-1) + \
                  torch.tensor(np.linspace(0,pred.shape[2]-1,pred.shape[2])).unsqueeze(0).repeat(pred.shape[0],1)
        _weight = _weight.unsqueeze(1).unsqueeze(-1)
        _weight = (_weight - (self.h / 2) + 0.5) * math.pi / self.h
        _weight = _weight.cos()
        _weight = _weight.repeat(1,pred.shape[1],1,pred.shape[3]).detach().cuda(pred.device)
        return self.loss_weight * l1_loss(pred, target, weight=_weight, reduction=self.reduction)
