# -*- coding: utf-8 -*-
"""Gate-1 model variants.

Two minimal BC policies sharing a frozen ResNet18 vision encoder.

VariantA — agentview RGB only:
    image (3,128,128) -> ResNet18 frozen -> 512-d feature
                                       -> MLP(512 -> 256 -> 7)

VariantB — agentview RGB + 15-d proprioception (ee_pos + ee_ori + joint + gripper):
    image -> ResNet18 frozen -> 512-d
    proprio (15) -> Linear(15 -> 64) -> 64-d
    concat -> 576-d
                                       -> MLP(576 -> 256 -> 7)

Both predict a single normalised action (7-dim).  Trained with MSE.
Variant B has only ~64K extra trainable parameters — so any success
delta is attributable to the *proprio signal*, not extra capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Normalize, Resize, Compose


# Image preprocessing matching torchvision's pre-trained ResNet18
_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]


def _make_image_encoder() -> tuple[nn.Module, int]:
    """ResNet18 ImageNet-pretrained, frozen, returning 512-d global-pool feature."""
    enc = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # strip final classifier
    feat_dim = enc.fc.in_features  # 512
    enc.fc = nn.Identity()
    # freeze
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc, feat_dim


class VariantA(nn.Module):
    """RGB-only baseline."""

    def __init__(self, action_dim: int = 7, hidden: int = 256, image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.encoder, feat = _make_image_encoder()
        self.preprocess = Compose([
            Resize((224, 224), antialias=True),
            Normalize(_IMG_MEAN, _IMG_STD),
        ])
        self.head = nn.Sequential(
            nn.Linear(feat, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        """image: (B,3,H,W) in [0,1]. proprio ignored (kept in API for parity)."""
        x = self.preprocess(image)
        with torch.no_grad():
            feat = self.encoder(x)                # (B, 512)
        return self.head(feat)


class VariantB(nn.Module):
    """RGB + proprio (poor-man's explicit 3D awareness)."""

    def __init__(self,
                 action_dim: int = 7,
                 proprio_dim: int = 15,
                 proprio_embed: int = 64,
                 hidden: int = 256,
                 image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.encoder, feat = _make_image_encoder()
        self.preprocess = Compose([
            Resize((224, 224), antialias=True),
            Normalize(_IMG_MEAN, _IMG_STD),
        ])
        self.proprio_enc = nn.Sequential(
            nn.Linear(proprio_dim, proprio_embed), nn.GELU(),
            nn.Linear(proprio_embed, proprio_embed),
        )
        self.head = nn.Sequential(
            nn.Linear(feat + proprio_embed, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(image)
        with torch.no_grad():
            v = self.encoder(x)
        p = self.proprio_enc(proprio)
        z = torch.cat([v, p], dim=-1)
        return self.head(z)


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
