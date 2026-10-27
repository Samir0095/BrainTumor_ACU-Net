import math
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




class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        pooled = self.avg_pool(x)
        pooled = pooled.view(batch_size, channels)

        weights = self.mlp(pooled)
        weights = self.sigmoid(weights)

        weights = weights.view(batch_size, channels, 1, 1)

        output = x * weights

        return output

class AttentionGate(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, inter_channels):
        super().__init__()

        self.encoder_projection = nn.Conv2d(
            encoder_channels, inter_channels, kernel_size=1, bias=False
        )
        self.decoder_projection = nn.Conv2d(
            decoder_channels, inter_channels, kernel_size=1, bias=False
        )

        self.sigmoid = nn.Sigmoid()
        self.scale = math.sqrt(inter_channels)

    def forward(self, encoder_features, decoder_features):
        keys = self.encoder_projection(encoder_features)
        queries = self.decoder_projection(decoder_features)

        scores = torch.sum(queries * keys, dim=1, keepdim=True) / self.scale
        attention_map = self.sigmoid(scores)

        gated_features = encoder_features * attention_map
        return gated_features