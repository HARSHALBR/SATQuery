"""
Sentinel-1 SAR (Dual-Polarization) Feature Encoder.

Processes 2-channel SAR imagery (VV and VH polarizations) through a lightweight
multi-stage residual convolutional hierarchy, extracting structural backscatter tokens.
"""

from typing import Optional

import torch
import torch.nn as nn


class SARResidualBlock(nn.Module):
    """Residual convolutional block with residual shortcut and Layer/Batch normalization."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class S1Encoder(nn.Module):
    """
    Sentinel-1 SAR Feature Encoder.
    
    Architecture Note:
        This is a task-specific initialized SAR modality encoder with a 4-stage
        residual convolutional stem. It is initialized with standard Gaussian weights
        and initially frozen for Step 2 forward pass verification.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            in_channels: Number of SAR channels (default: 2 for VV and VH).
            hidden_dim: Output token embedding dimension (default: 512).
            freeze_backbone: If True, freezes all parameters during initialization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Stage 1: Initial Stem (stride 2) -> [B, 64, H/2, W/2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Stage 2: Residual Block (stride 2) -> [B, 128, H/4, W/4]
        self.stage2 = SARResidualBlock(64, 128, stride=2)

        # Stage 3: Residual Block (stride 2) -> [B, 256, H/8, W/8]
        self.stage3 = SARResidualBlock(128, 256, stride=2)

        # Stage 4: Projection head -> [B, hidden_dim, H/8, W/8]
        self.head = nn.Sequential(
            nn.Conv2d(256, hidden_dim, kernel_size=1, stride=1, bias=True),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

        if freeze_backbone:
            self.freeze()

    def _init_weights(self) -> None:
        """Initialize weights using standard Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze(self) -> None:
        """Freeze all parameters in the S1 encoder."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters in the S1 encoder for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Sentinel-1 tensor of shape [B, in_channels, H, W] (e.g. [B, 2, 120, 120]).
            
        Returns:
            S1 visual tokens of shape [B, N_s1, hidden_dim] where N_s1 = (H/8)*(W/8) (e.g. 225 tokens for 120x120).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels for S1, got {x.shape[1]}"
            )

        # Convolutional stages: 8x downsampling
        h = self.stem(x)        # [B, 64, H/2, W/2]
        h = self.stage2(h)      # [B, 128, H/4, W/4]
        h = self.stage3(h)      # [B, 256, H/8, W/8]
        h = self.head(h)        # [B, hidden_dim, H/8, W/8]

        # Reshape to token sequence: [B, C, H', W'] -> [B, H'*W', C]
        B, C, H_prime, W_prime = h.shape
        tokens = h.flatten(2).transpose(1, 2)  # [B, N_s1, hidden_dim]
        tokens = self.norm(tokens)

        return tokens
