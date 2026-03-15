from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.losses._utils import mask_ignore_pixels

def focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float=0.7, # Fasse negative panalty
    beta: float=0.3, # False positive penalty
    gamma: float=1.33, # Focal term exponent, controls the down-weighting of easy examples
    eps: float = 1e-8, # Small constant to avoid division by zero
    ignore_index: Optional[int] = -100,
) -> torch.Tensor:
    '''
    Implemantation based on tvensky_loss from kornia, but with the focal term added as in https://arxiv.org/pdf/1810.07842.pdf
    '''

    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}")

    if not len(pred.shape) == 4:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW. Got: {pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    # compute softmax over the classes axis
    pred_soft = F.softmax(pred, dim=1)
    target, target_mask = mask_ignore_pixels(target, ignore_index)

    p_true = pred_soft.gather(1, target.unsqueeze(1))  # (B,1,H,W)

    if target_mask is not None:
        m = target_mask.unsqueeze(1).to(dtype=pred.dtype)
        p_true = p_true * m
        total = m.sum((1, 2, 3))
    else:
        B, _, H, W = pred.shape
        total = torch.full((B,), H * W, dtype=pred.dtype, device=pred.device)

    intersection = p_true.sum((1, 2, 3))
    # denominator = intersection + (alpha + beta) * (total - intersection) + eps
    # instead of multiple ops, do it in one fused step:
    denominator = torch.addcmul(
        intersection,  # base
        total - intersection,  # tensor1
        torch.full_like(total, alpha + beta),  # tensor2 (scalar as tensor)
        value=1.0,  # (intersection) + 1 * (tensor1*tensor2)
    ).add_(eps)  # in-place add eps
    score = intersection.div(denominator)

    return torch.pow(1.0 - score, gamma).mean()


class FocalTverskyLoss(nn.Module):

    def __init__(self, alpha: float=0.7, beta: float=0.3, gamma: float=1.33, eps: float = 1e-8, ignore_index: Optional[int] = -100) -> None:
        super().__init__()
        self.alpha: float = alpha # False negative penalty
        self.beta: float = beta # False positive penalty
        self.gamma: float = gamma # Focal term exponent, controls the down-weighting of easy examples
        self.eps: float = eps # Small constant to avoid division by zero
        self.ignore_index: Optional[int] = ignore_index # Class index to ignore in the loss computation, typically used for padding or unlabeled pixels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_tversky_loss(pred, target, self.alpha, self.beta, self.gamma, self.eps, self.ignore_index)