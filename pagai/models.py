"""
Channel-stacked 3D vision encoders. The input [B, C, X, Y, Z] (C = 3: T2/DWI/ADC, or 4 with the zone
mask) goes through one trunk; forward() returns its token sequence [B, N, embed_dim], which the
classifier mean-pools.

    vit       MONAI 3D ViT (conv patch embedding, learnable position embedding, no cls token)
    resnet18  MONAI 3D ResNet-18, layer4 feature map flattened to tokens (embed_dim 512)
    resnet50  MONAI 3D ResNet-50, same (embed_dim 2048)

All are trained from scratch. Attribute names match the original research code so its checkpoints
load unchanged (see checkpoint.py).
"""

import torch.nn as nn
from monai.networks.nets import ViT, resnet18, resnet50

MODEL_CHOICES = ("vit", "resnet18", "resnet50")
# --ve_name values of the original research code -> this repo's --model.
LEGACY_MODEL_NAMES = {"3dvit_channel": "vit", "resnet18_channel": "resnet18", "resnet50_channel": "resnet50"}

VIT_DEFAULTS = {
    "vit_patch_size": (16, 16, 8),
    "vit_hidden_size": 768,
    "vit_mlp_dim": 3072,
    "vit_num_layers": 12,
    "vit_num_heads": 12,
    "vit_dropout_rate": 0.1,
}


class _SizeCheck(nn.Module):
    def _check_size(self, x):
        if tuple(x.shape[-3:]) != self.img_size:
            raise ValueError(
                f"input has spatial shape {tuple(x.shape[-3:])}, expected {self.img_size} (--img_size); "
                f"{type(self).__name__} does not resize"
            )


class ViT3DChannelBackbone(_SizeCheck):
    def __init__(self, img_size, patch_size, in_channels=3, hidden_size=768, mlp_dim=3072, num_layers=12,
                 num_heads=12, dropout_rate=0.0):
        super().__init__()
        self.img_size = tuple(img_size)
        self.embed_dim = hidden_size
        if any(s % p for s, p in zip(self.img_size, patch_size)):
            raise ValueError(f"--img_size {self.img_size} must be divisible by --vit_patch_size {tuple(patch_size)}")

        self.vit = ViT(
            in_channels=in_channels,
            img_size=self.img_size,
            patch_size=tuple(patch_size),
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type="conv",
            pos_embed_type="learnable",
            classification=False,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )
        # MONAI builds a cross-attention LayerNorm per block that forward() never uses (no cross
        # attention here); freeze it so it doesn't look like a trainable-but-unused parameter.
        for block in self.vit.blocks:
            for p in block.norm_cross_attn.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        self._check_size(x)
        tokens, _ = self.vit(x)  # [B, N, hidden_size]
        return tokens


class ResNetChannelBackbone(_SizeCheck):
    def __init__(self, img_size, depth=18, in_channels=3):
        super().__init__()
        self.img_size = tuple(img_size)
        if depth == 18:
            resnet, self.embed_dim = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=1), 512
        elif depth == 50:
            resnet, self.embed_dim = resnet50(spatial_dims=3, n_input_channels=in_channels, num_classes=1), 2048
        else:
            raise ValueError(f"unsupported ResNet depth {depth}")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.act = resnet.act
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        self._check_size(x)
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))  # [B, embed_dim, X/16, Y/16, Z/16]
        return x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]


def build_backbone(model, img_size, in_channels, vit_kwargs=None):
    """vit_kwargs: the VIT_DEFAULTS keys (only used for model='vit'); missing keys take the defaults."""
    if model == "vit":
        cfg = {**VIT_DEFAULTS, **(vit_kwargs or {})}
        return ViT3DChannelBackbone(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=cfg["vit_patch_size"],
            hidden_size=cfg["vit_hidden_size"],
            mlp_dim=cfg["vit_mlp_dim"],
            num_layers=cfg["vit_num_layers"],
            num_heads=cfg["vit_num_heads"],
            dropout_rate=cfg["vit_dropout_rate"],
        )
    if model in ("resnet18", "resnet50"):
        return ResNetChannelBackbone(img_size=img_size, depth=int(model[len("resnet"):]), in_channels=in_channels)
    raise ValueError(f"unknown model {model!r}; choose from {MODEL_CHOICES}")
