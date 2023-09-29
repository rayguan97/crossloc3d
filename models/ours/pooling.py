import torch
import torch.nn as nn
import torch.nn.functional as F


class MAC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        return torch.max(x, dim=-1, keepdim=False)[0]


class SPoC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        return torch.mean(x, dim=-1, keepdim=False)


class GeM(nn.Module):
    def __init__(self, cfg):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * cfg.p)
        self.eps = cfg.eps

    def forward(self, x):
        # This implicitly applies ReLU on x (clamps negative values)
        x = x.clamp(min=self.eps).pow(self.p)
        # Apply ME.MinkowskiGlobalAvgPooling
        x = torch.max(x, dim=-1, keepdim=False)
        # Return (batch_size, n_features) tensor
        return x.pow(1./self.p)
