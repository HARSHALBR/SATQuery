"""
Sentinel-2 Multispectral (10-Band: 10m & 20m) Feature Encoder.

Processes 10-channel multispectral optical imagery (B02-B12, excluding 60m bands B01 and B09)
through a multispectral patch embedding stem and transformer encoder blocks.
"""

from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder block with multi-head self-attention and MLP."""

    def __init__(self, dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm & residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # MLP with pre-norm & residual
        x = x + self.mlp(self.norm2(x))
        return x


class S2Encoder(nn.Module):
    """
    Sentinel-2 Multispectral Feature Encoder.
    
    Architecture Note:
        This is a task-specific initialized multispectral modality encoder tailored
        for the 10 BigEarthNet 10m and 20m optical bands. It is initialized with
        standard normal weights and initially frozen for Step 2 forward pass verification.
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 768,
        patch_size: int = 8,
        img_size: int = 120,
        num_layers: int = 2,
        num_heads: int = 8,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            in_channels: Number of optical channels (default: 10 for 10m + 20m bands).
            hidden_dim: Output token embedding dimension (default: 768).
            patch_size: Spatial patch size for embedding stem (default: 8).
            img_size: Expected spatial resolution (default: 120).
            num_layers: Number of transformer blocks (default: 2).
            num_heads: Attention heads (default: 8).
            freeze_backbone: If True, freezes all parameters during initialization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.img_size = img_size

        self.num_patches = (img_size // patch_size) ** 2  # (120/8)^2 = 225

        # Multispectral Patch Embedding Stem
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Learnable 1D/2D Spatial Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

        if freeze_backbone:
            self.freeze()

    def _init_weights(self) -> None:
        """Initialize weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze(self) -> None:
        """Freeze all parameters in the S2 encoder."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters in the S2 encoder for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Sentinel-2 tensor of shape [B, in_channels, H, W] (e.g. [B, 10, 120, 120]).
            
        Returns:
            S2 visual tokens of shape [B, N_s2, hidden_dim] where N_s2 = (H/8)*(W/8) (e.g. 225 tokens for 120x120).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels for S2, got {x.shape[1]}"
            )

        # Patch embedding: [B, C, H, W] -> [B, hidden_dim, H/8, W/8]
        h = self.patch_embed(x)
        B, C, H_prime, W_prime = h.shape
        num_patches = H_prime * W_prime

        # Flatten into tokens: [B, num_patches, hidden_dim]
        tokens = h.flatten(2).transpose(1, 2)

        # Add positional embeddings (interpolating if resolution differs)
        if tokens.shape[1] == self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed
        else:
            # Linear interpolation for variable patch counts
            pos = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            tokens = tokens + pos

        # Apply transformer encoder blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens
