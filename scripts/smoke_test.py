from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from s1_test_time_scaling_unofficial import ModelConfig, UnofficialModel, reconstruction_loss


def main() -> None:
    torch.manual_seed(2026)
    config = ModelConfig(task="reasoning", hidden_dim=64, num_layers=2, num_heads=4, output_dim=64)
    model = UnofficialModel(config)
    image = torch.rand(2, 3, 64, 64)
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    out = model(image, token_ids=token_ids)
    loss = reconstruction_loss(out.primary)
    loss.backward()
    print(f"output: {tuple(out.primary.shape)}")
    print(f"loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
