"""Unit tests: QualityImprover (noise reduction helpers)."""

import numpy as np
import pytest

from app.core.QualityImprover import QualityImprover


class TestMedianBlur:
    def test_positive_default_ksize(self, sample_rgb_frame):
        out = QualityImprover.medianBlur(sample_rgb_frame.copy(), ksize=3)
        assert out.shape == sample_rgb_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("bad_ksize,expected_odd", [(2, 3), (1, 3), (4, 5)])
    def test_even_or_small_ksize_normalized(self, sample_rgb_frame, bad_ksize, expected_odd):
        out = QualityImprover.medianBlur(sample_rgb_frame.copy(), ksize=bad_ksize)
        assert out.shape == sample_rgb_frame.shape


class TestFastGaussian:
    def test_positive(self, sample_rgb_frame):
        out = QualityImprover.fast_gaussian(sample_rgb_frame.copy(), ksize=5, sigma=1.2)
        assert out.shape == sample_rgb_frame.shape

    def test_small_ksize_becomes_odd(self, sample_rgb_frame):
        out = QualityImprover.fast_gaussian(sample_rgb_frame.copy(), ksize=2, sigma=0.5)
        assert out.shape == sample_rgb_frame.shape
