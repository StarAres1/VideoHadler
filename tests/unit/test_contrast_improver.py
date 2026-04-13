"""Unit tests: ContrastImprover static methods."""

import numpy as np
import pytest

from app.core.ContrastImprover import ContrastImprover


@pytest.fixture
def rgb_square():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = 50
    img[:, :, 1] = 100
    img[:, :, 2] = 200
    return img


class TestCLAHE:
    def test_positive_color(self, rgb_square):
        out = ContrastImprover.CLAHE(rgb_square, clipLimit=2.0, titleGridSizeX=4, titleGridSizeY=4)
        assert out.shape == rgb_square.shape
        assert out.dtype == np.uint8


class TestHE:
    def test_positive_color(self, rgb_square):
        out = ContrastImprover.HE(rgb_square)
        assert out.shape == rgb_square.shape


class TestAdjustContrast:
    def test_positive(self, rgb_square):
        out = ContrastImprover.adjust_contrast(rgb_square, alpha=1.2, beta=5)
        assert out.shape == rgb_square.shape


class TestGammaCorrection:
    def test_positive(self, rgb_square):
        out = ContrastImprover.gamma_correction(rgb_square, gamma=1.5)
        assert out.shape == rgb_square.shape


class TestSigmoidCorrection:
    def test_positive(self, rgb_square):
        out = ContrastImprover.sigmoid_correction(rgb_square, cutoff=0.5, gain=8)
        assert out.shape == rgb_square.shape


class TestAutoGamma:
    def test_positive_color(self, rgb_square):
        out = ContrastImprover.auto_gamma(rgb_square, target_brightness=128)
        assert out.shape == rgb_square.shape
