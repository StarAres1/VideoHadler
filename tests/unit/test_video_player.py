"""Unit tests: VideoPlayer (Qt + OpenCV)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel

from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.VideoPlayer import VideoPlayer


@pytest.fixture
def label_record_format(qtbot):
    w = QLabel()
    qtbot.addWidget(w)
    return w


class TestVideoPlayerBasics:
    def test_initial_not_loaded(self, label_record_format):
        vp = VideoPlayer(label_record_format)
        assert vp.is_loaded() is False
        assert vp.is_playing() is False

    def test_negative_play_bad_path(self, label_record_format, tmp_path):
        vp = VideoPlayer(label_record_format)
        bad = str(tmp_path / "missing.mp4")
        vp.play(bad)
        assert vp.cap is not None
        assert not vp.cap.isOpened()

    def test_positive_play_real_file(self, label_record_format, video_file_mp4, qtbot):
        vp = VideoPlayer(label_record_format)
        vp.play(video_file_mp4)
        assert vp.is_loaded()
        assert vp.width > 0 and vp.height > 0
        vp.stop()
        qtbot.wait(50)

    def test_toggle_pause_resume(self, label_record_format, video_file_mp4, qtbot):
        vp = VideoPlayer(label_record_format)
        vp.play(video_file_mp4)
        assert vp.is_playing() is True
        vp.toggle_play_pause()
        assert vp.is_playing() is False
        vp.toggle_play_pause()
        assert vp.is_playing() is True
        vp.stop()
        qtbot.wait(50)

    def test_seek_seconds_clamped(self, label_record_format, video_file_mp4):
        vp = VideoPlayer(label_record_format)
        vp.play(video_file_mp4)
        vp.seek_seconds(-999999)
        vp.stop()

    def test_set_position_percent(self, label_record_format, video_file_mp4):
        vp = VideoPlayer(label_record_format)
        vp.play(video_file_mp4)
        vp.set_position_percent(50)
        vp.stop()

    def test_format_time(self, label_record_format):
        vp = VideoPlayer(label_record_format)
        assert vp._format_time(3661000) == "1:01:01"

    def test_save_screenshot_negative(self, label_record_format):
        vp = VideoPlayer(label_record_format)
        assert vp.save_screenshot() is None

    def test_process_frame_none_returns_none(self, label_record_format):
        vp = VideoPlayer(label_record_format)
        p = vp._snapshot_params()
        assert vp._process_frame(None, p) is None

    def test_snapshot_params(self, label_record_format):
        vp = VideoPlayer(label_record_format)
        vp.width = 10
        vp.height = 20
        vp.roi_x = 1
        vp.roi_y = 2
        vp.roi_width = 5
        vp.roi_height = 6
        vp.method_for_noise = NoiseReduction.MedianBlur
        vp.method_for_contrast = ContrastImprovement.gamma
        s = vp._snapshot_params()
        assert s["width"] == 10
        assert s["noise"] == NoiseReduction.MedianBlur

    def test_set_methods_update_state(self, label_record_format):
        vp = VideoPlayer(label_record_format)
        vp.set_method_for_contrast(ContrastImprovement.CLAHE)
        assert vp.method_for_contrast == ContrastImprovement.CLAHE
        vp.set_method_for_noise(NoiseReduction.FastGaussian)
        assert vp.method_for_noise == NoiseReduction.FastGaussian
