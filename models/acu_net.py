
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]

        combined = torch.cat([avg_map, max_map], dim=1)

        attention_map = self.sigmoid(self.conv(combined))

        output = x * attention_map

        return output
