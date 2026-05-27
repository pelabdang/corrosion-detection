"""U-Net model for steel surface defect segmentation."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        num_classes: Number of output segmentation classes.
        features: List of feature sizes for each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        features: list[int] | None = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()

        # Encoder path
        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(len(self.decoder_blocks)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )

            x = torch.cat((skip, x), dim=1)
            x = self.decoder_blocks[idx](x)

        return self.final_conv(x)
