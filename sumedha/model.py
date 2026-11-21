"""Residual convolutional policy/value network for chess."""
from __future__ import annotations

import math
import torch
from torch import nn

from .encoding import INPUT_PLANES, POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class SumedhaNet(nn.Module):
    """Outputs unmasked policy logits [B, 20480] and side-to-move value [B]."""

    def __init__(self, channels: int = 96, blocks: int = 6, embedding: int = 64):
        super().__init__()
        if channels < 16 or blocks < 1 or embedding < 8:
            raise ValueError("Model dimensions are too small.")
        self.config = {
            "channels": int(channels),
            "blocks": int(blocks),
            "embedding": int(embedding),
            "input_planes": INPUT_PLANES,
            "policy_size": POLICY_SIZE,
        }
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Factorized from/to scoring avoids a huge fully-connected policy layer.
        self.move_embed = nn.Conv2d(channels, embedding * 2 + 4, kernel_size=1)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(32 * 64, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(x))
        batch = x.shape[0]
        per_square = self.move_embed(x).permute(0, 2, 3, 1).reshape(batch, 64, -1)
        embedding = self.config["embedding"]
        query = per_square[:, :, :embedding]
        key = per_square[:, :, embedding:2 * embedding]
        promo = per_square[:, :, 2 * embedding:]  # [B, destination-square, 4]

        pair_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(embedding)
        promo_logits = promo.unsqueeze(1).expand(-1, 64, -1, -1)
        policy = torch.cat((pair_logits.unsqueeze(-1), pair_logits.unsqueeze(-1) + promo_logits), dim=-1)
        policy = policy.reshape(batch, POLICY_SIZE)

        value_features = self.value_conv(x).reshape(batch, -1)
        value = self.value_mlp(value_features).squeeze(-1)
        return policy, value
