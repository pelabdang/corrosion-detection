"""Unit tests for the U-Net model."""

import torch
import pytest

from src.model import UNet, DoubleConv


class TestDoubleConv:
    """Tests for the DoubleConv block."""

    def test_output_shape(self):
        block = DoubleConv(3, 64)
        x = torch.randn(2, 3, 100, 100)
        output = block(x)
        assert output.shape == (2, 64, 100, 100)

    def test_different_sizes(self):
        block = DoubleConv(64, 128)
        x = torch.randn(1, 64, 50, 50)
        output = block(x)
        assert output.shape == (1, 128, 50, 50)


class TestUNet:
    """Tests for the U-Net model."""

    def test_default_output_shape(self):
        model = UNet(in_channels=3, num_classes=7)
        x = torch.randn(1, 3, 200, 200)
        output = model(x)
        assert output.shape == (1, 7, 200, 200)

    def test_grayscale_input(self):
        model = UNet(in_channels=1, num_classes=2)
        x = torch.randn(1, 1, 128, 128)
        output = model(x)
        assert output.shape == (1, 2, 128, 128)

    def test_batch_processing(self):
        model = UNet(in_channels=3, num_classes=7)
        x = torch.randn(4, 3, 200, 200)
        output = model(x)
        assert output.shape == (4, 7, 200, 200)

    def test_non_square_input(self):
        model = UNet(in_channels=3, num_classes=4)
        x = torch.randn(1, 3, 192, 256)
        output = model(x)
        assert output.shape == (1, 4, 192, 256)

    def test_custom_features(self):
        model = UNet(in_channels=3, num_classes=3, features=[32, 64, 128, 256])
        x = torch.randn(1, 3, 200, 200)
        output = model(x)
        assert output.shape == (1, 3, 200, 200)

    def test_gradient_flow(self):
        model = UNet(in_channels=3, num_classes=7)
        x = torch.randn(1, 3, 200, 200, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameter_count(self):
        model = UNet(in_channels=3, num_classes=7)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        # U-Net with default features should have ~31M parameters
        assert total_params > 1_000_000
