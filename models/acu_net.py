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




class DualAttention(nn.Module):
    def __init__(self, channels, spatial_kernel_size=7, reduction=16):
        super().__init__()

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size
        )

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction
        )

    def forward(self, x):
        spatial_features = self.spatial_attention(x)
        channel_features = self.channel_attention(x)

        output = spatial_features + channel_features

        return output



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



class ACUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(4, 16)
        self.encoder2 = DoubleConv(16, 32)
        self.encoder3 = DoubleConv(32, 64)
        self.encoder4 = DoubleConv(64, 128)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dual Attention at the bottleneck
        self.dual_attention = DualAttention(channels=256)

        # Decoder upsampling
        self.upconv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.upconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.upconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.upconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2
        )

        # Attention Gates for skip connections
        self.attention_gate4 = AttentionGate(
            encoder_channels=128,
            decoder_channels=128,
            inter_channels=64
        )
        self.attention_gate3 = AttentionGate(
            encoder_channels=64,
            decoder_channels=64,
            inter_channels=32
        )
        self.attention_gate2 = AttentionGate(
            encoder_channels=32,
            decoder_channels=32,
            inter_channels=16
        )
        self.attention_gate1 = AttentionGate(
            encoder_channels=16,
            decoder_channels=16,
            inter_channels=8
        )

        # Decoder convolution blocks
        self.decoder4 = DoubleConv(256, 128)
        self.decoder3 = DoubleConv(128, 64)
        self.decoder2 = DoubleConv(64, 32)
        self.decoder1 = DoubleConv(32, 16)

        # Raw segmentation logits
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck + Dual Attention
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dual_attention(bottleneck)

        # Decoder stage 4
        dec4 = self.upconv4(bottleneck)
        gated_enc4 = self.attention_gate4(enc4, dec4)
        dec4 = torch.cat([dec4, gated_enc4], dim=1)
        dec4 = self.decoder4(dec4)

        # Decoder stage 3
        dec3 = self.upconv3(dec4)
        gated_enc3 = self.attention_gate3(enc3, dec3)
        dec3 = torch.cat([dec3, gated_enc3], dim=1)
        dec3 = self.decoder3(dec3)

        # Decoder stage 2
        dec2 = self.upconv2(dec3)
        gated_enc2 = self.attention_gate2(enc2, dec2)
        dec2 = torch.cat([dec2, gated_enc2], dim=1)
        dec2 = self.decoder2(dec2)

        # Decoder stage 1
        dec1 = self.upconv1(dec2)
        gated_enc1 = self.attention_gate1(enc1, dec1)
        dec1 = torch.cat([dec1, gated_enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return self.output(dec1)
