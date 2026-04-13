"""Unit tests: FrameProcessor and ProcessingConfig."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.FrameProcessor import FrameProcessor, ProcessingConfig


@pytest.fixture
def rgb_frame():
    return np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)


class TestProcessingConfig:
    def test_defaults(self):
        c = ProcessingConfig()
        assert c.clip_limit == 2.0
        assert c.tile_grid_size == 4
        assert c.monochrome is False


class TestFrameProcessorProcess:
    def test_not_improve_pass_through_shape(self, rgb_frame):
        fp = FrameProcessor()
        out = fp.process(
            rgb_frame.copy(),
            30,
            20,
            0,
            0,
            30,
            20,
            ContrastImprovement.NotImprove,
            NoiseReduction.NotReduction,
        )
        assert out.shape == (20, 30, 3)

    def test_median_blur_positive(self, rgb_frame):
        fp = FrameProcessor()
        out = fp.process(
            rgb_frame,
            30,
            20,
            0,
            0,
            30,
            20,
            ContrastImprovement.NotImprove,
            NoiseReduction.MedianBlur,
        )
        assert out.shape == (20, 30, 3)

    def test_fast_gaussian_positive(self, rgb_frame):
        fp = FrameProcessor()
        out = fp.process(
            rgb_frame,
            30,
            20,
            0,
            0,
            30,
            20,
            ContrastImprovement.NotImprove,
            NoiseReduction.FastGaussian,
        )
        assert out.shape == (20, 30, 3)

    def test_clahe_positive(self, rgb_frame):
        fp = FrameProcessor()
        out = fp.process(
            rgb_frame,
            30,
            20,
            0,
            0,
            30,
            20,
            ContrastImprovement.CLAHE,
            NoiseReduction.NotReduction,
        )
        assert out.shape == (20, 30, 3)

    def test_monochrome_toggle(self, rgb_frame):
        fp = FrameProcessor(ProcessingConfig(monochrome=True))
        out = fp.process(
            rgb_frame,
            30,
            20,
            0,
            0,
            30,
            20,
            ContrastImprovement.NotImprove,
            NoiseReduction.NotReduction,
        )
        assert out.shape == (20, 30, 3)

    def test_roi_clamping_negative_extreme(self, rgb_frame):
        fp = FrameProcessor()
        out = fp.process(
            rgb_frame,
            30,
            20,
            999,
            999,
            999,
            999,
            ContrastImprovement.NotImprove,
            NoiseReduction.NotReduction,
        )
        assert out.shape == (20, 30, 3)

    def test_nn_branch_with_mock_selector(self, rgb_frame):
        mock_nn = MagicMock()
        mock_nn.predict_label.return_value = "gamma_1.5"
        mock_nn.apply_label.side_effect = lambda frame, label_record_format: frame

        fp = FrameProcessor()
        fp.nn_selector = mock_nn
        fp.config.nn_skip_frames = 0

        out = fp.process(
            rgb_frame,
            30,
            20,
            0,
            0,
            30,
            20,
            ContrastImprovement.nn,
            NoiseReduction.NotReduction,
        )
        assert out.shape == (20, 30, 3)
        mock_nn.predict_label.assert_called()

    def test_nn_skip_frames_uses_cache(self, rgb_frame):
        mock_nn = MagicMock()
        mock_nn.predict_label.return_value = "gamma_1.2"
        mock_nn.apply_label.side_effect = lambda f, _: f

        fp = FrameProcessor()
        fp.nn_selector = mock_nn
        fp.config.nn_skip_frames = 2

        fp.process(
            rgb_frame, 30, 20, 0, 0, 30, 20, ContrastImprovement.nn, NoiseReduction.NotReduction
        )
        fp.process(
            rgb_frame, 30, 20, 0, 0, 30, 20, ContrastImprovement.nn, NoiseReduction.NotReduction
        )
        fp.process(
            rgb_frame, 30, 20, 0, 0, 30, 20, ContrastImprovement.nn, NoiseReduction.NotReduction
        )
        assert mock_nn.predict_label.call_count >= 1
