# MIT License
#
# Copyright (c) 2021 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


def undo_squashed_scale(
    x, clip_soft=384, track_transform=3 / 4, track_scale=0.01, old_transform=True
):
    """
    Reverses the squashed scaling transformation applied to the output profiles.

    Args:
        x (torch.Tensor): The input tensor to be unsquashed.
        clip_soft (float, optional): The soft clipping value. Defaults to 384.
        track_transform (float, optional): The transformation factor. Defaults to 3/4.
        track_scale (float, optional): The scale factor. Defaults to 0.01.

    Returns:
        torch.Tensor: The unsquashed tensor.
    """
    x = x.clone()  # IMPORTANT BECAUSE OF IMPLACE OPERATIONS TO FOLLOW?

    if old_transform:
        x = x / track_scale
        unclip_mask = x > clip_soft
        x[unclip_mask] = (x[unclip_mask] - clip_soft) ** 2 + clip_soft
        x = x ** (1.0 / track_transform)
    else:
        unclip_mask = x > clip_soft
        x[unclip_mask] = (x[unclip_mask] - clip_soft + 1) ** 2 + clip_soft - 1
        x = (x + 1) ** (1.0 / track_transform) - 1
        x = x / track_scale
    return x


class PoissonMultinomial(nn.Module):
    """
    Poisson loss decomposition with a multinomial term
    Adapted from https://github.com/johahi/training-borzoi
    """

    def __init__(
        self,
        total_weight: float = 0.2,
        epsilon: float = 1e-6,
        rescale: bool = False
    ):
        super().__init__()
        self.total_weight = total_weight
        self.epsilon = epsilon
        self.rescale = rescale

    def forward(self, y_pred, y_true):

        seq_len = y_true.shape[1]

        # epsilon protects against tiny/zero values
        y_pred += self.epsilon
        y_true += self.epsilon

        # sum across lengths to normalise
        s_pred = y_pred.sum(dim=1, keepdim=True)
        s_true = y_true.sum(dim=1, keepdim=True)

        # normalise to sum to one
        p_pred = y_pred / s_pred
        
        poisson_term = F.poisson_nll_loss(
            s_pred, s_true, log_input=False, eps=0, reduction="mean"
        )  # B x T

        poisson_term /= seq_len

        # multinomial loss
        pl_pred = torch.log(p_pred)  # B x L x T
        multinomial_dot = -torch.multiply(y_true, pl_pred)  # B x L x T
        multinomial_term = multinomial_dot.sum(dim=1)  # B x T
        multinomial_term /= seq_len

        # normalize to scale of 1:1 term ratio
        loss_raw = multinomial_term + self.total_weight * poisson_term
        if self.rescale:
            loss_rescale = loss_raw * 2 / (1 + self.total_weight)
        else:
            loss_rescale = loss_raw

        return loss_rescale.mean()
       

class SparseMSELoss(nn.Module):
    """
    Loss to handle predictions of beta values in sparse sequences.
    Combines BCE loss to find non-zero positions then applies MSE loss.
    """
    def __init__(self, bce_weight: float = 1.0, mse_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.eps = eps # threshold to count as non-zero

    def forward(self, y_pred, y_true):
        # Is the site non-zero?
        presence_true = (y_true > self.eps).float()

         # BCE loss component
        bce_loss = F.binary_cross_entropy(y_pred, presence_true)
        
        # Masked MSE loss for value at non-zero positions
        mask = presence_true.bool()
        if mask.any():
            mse_loss = F.mse_loss(y_pred[mask], y_true[mask])
        else:
            mse_loss = torch.tensor(0.0, device=y_pred.device)
        
        # Combine loss terms
        loss = self.bce_weight * bce_loss + self.mse_weight * mse_loss
        return loss