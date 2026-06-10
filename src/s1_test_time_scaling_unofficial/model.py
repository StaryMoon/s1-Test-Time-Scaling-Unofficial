from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    task: str = "multimodal"
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    vocab_size: int = 32000
    output_dim: int = 128
    action_dim: int = 7


@dataclass
class ModelOutput:
    primary: torch.Tensor
    features: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        key_value = x if context is None else torch.cat([context, x], dim=1)
        y, _ = self.attn(self.norm(x), self.norm(key_value), self.norm(key_value))
        x = x + y
        return x + self.ffn(x)


class UnofficialModel(nn.Module):
    """Compact PyTorch interface for unofficial reproduction experiments."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__()
        self.config = config or ModelConfig(**kwargs)
        hidden = self.config.hidden_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8 if (hidden // 2) % 8 == 0 else 1, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(),
        )
        self.text_embedding = nn.Embedding(self.config.vocab_size, hidden)
        self.layers = nn.ModuleList([ResidualBlock(hidden, self.config.num_heads) for _ in range(self.config.num_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.vector_head = nn.Linear(hidden, self.config.output_dim)
        self.image_head = nn.Conv2d(hidden, 3, 1)
        self.action_head = nn.Linear(hidden, self.config.action_dim)

    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature = self.image_encoder(image)
        size = feature.shape[-2:]
        tokens = feature.flatten(2).transpose(1, 2)
        return tokens, size

    def encode_text(self, token_ids: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if token_ids is None:
            token_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        return self.text_embedding(token_ids)

    def forward(self, image: torch.Tensor, token_ids: torch.Tensor | None = None) -> ModelOutput:
        tokens, size = self.encode_image(image)
        context = self.encode_text(token_ids, image.shape[0], image.device)
        for layer in self.layers:
            tokens = layer(tokens, context=context)
        pooled = self.norm(tokens.mean(dim=1))

        if self.config.task in {"generation", "video", "world", "3d"}:
            fmap = tokens.transpose(1, 2).reshape(image.shape[0], self.config.hidden_dim, *size)
            image_like = self.image_head(fmap)
            image_like = F.interpolate(image_like, size=image.shape[-2:], mode="bilinear", align_corners=False)
            return ModelOutput(primary=image_like, features=tokens)
        if self.config.task in {"robot", "vla"}:
            return ModelOutput(primary=self.action_head(pooled), features=tokens)
        return ModelOutput(primary=self.vector_head(pooled), features=tokens)


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    if target is None or target.shape != prediction.shape:
        target = torch.zeros_like(prediction)
    return F.smooth_l1_loss(prediction, target)
