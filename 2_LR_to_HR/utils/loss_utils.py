import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lat_np(arr, num_lat):
    return np.arcsin(2 * arr / num_lat - 1) * (180 / np.pi)

def latitude_weighting_factor(latitudes, num_lat, s):
    return np.cos(np.pi / 180 * lat_np(latitudes, num_lat)) / s

def weighted_charbonnier_loss(pred, target, epsilon=1e-3):
    num_lat = pred.shape[-2]
    num_lon = pred.shape[-1]
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = torch.tensor(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), dtype=pred.dtype, device=pred.device).view(1, 1, -1, 1)
    return (weight * torch.sqrt((pred - target) ** 2 + epsilon ** 2)).mean()

class NudgingLoss(nn.Module):
    def __init__(self, hr_weight=.5, lr_weight=.2, epsilon=1e-3):
        super(NudgingLoss, self).__init__()
        self.hr_weight = hr_weight
        self.lr_weight = lr_weight
        self.epsilon = epsilon

    def forward(self, pred, target, hr_pred, hr_target, lr_pred, lr_target):
        main_loss = weighted_charbonnier_loss(
            pred,
            target,
            epsilon=self.epsilon
        )
        hr_loss = weighted_charbonnier_loss(
            hr_pred,
            hr_target,
            epsilon=self.epsilon
        )
        lr_loss = weighted_charbonnier_loss(
            lr_pred,
            lr_target,
            epsilon=self.epsilon
        )
        return main_loss + self.hr_weight * hr_loss + self.lr_weight * lr_loss, main_loss, hr_loss, lr_loss