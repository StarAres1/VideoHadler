"""Unit tests: Recorder (video writer wiring)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from app.core.Recorder import Recorder


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Recorder()


class TestRecorderOpenFile:
    def test_positive_avi(self, recorder):
        with patch("app.core.Recorder.cv2.VideoWriter") as vw:
            mock_writer = MagicMock()
            vw.return_value = mock_writer
            recorder.open_file(320, 240, 30.0, "Cam One", "avi")
            assert recorder.video_format == "avi"
            assert recorder.width == 320
            vw.assert_called_once()

    def test_positive_mp4_fourcc(self, recorder):
        with patch("app.core.Recorder.cv2.VideoWriter") as vw:
            vw.return_value = MagicMock()
            recorder.open_file(64, 64, 0.0, "X", "mp4")
            assert recorder.video_format == "mp4"
            assert recorder.fps == 30.0

    def test_negative_none_format_defaults_avi(self, recorder):
        with patch("app.core.Recorder.cv2.VideoWriter") as vw:
            vw.return_value = MagicMock()
            recorder.open_file(10, 10, 10.0, "N", None)
            assert recorder.video_format == "avi"


class TestRecorderRecord:
    def test_positive_writes_when_output(self, recorder):
        with patch("app.core.Recorder.cv2.VideoWriter") as vw:
            mock_out = MagicMock()
            vw.return_value = mock_out
            recorder.open_file(4, 4, 10.0, "T", "avi")
            img = QImage(4, 4, QImage.Format.Format_RGB888)
            img.fill(0)
            recorder.record(img)
            mock_out.write.assert_called()

    def test_negative_no_output_skips(self, recorder):
        recorder.output = None
        img = QImage(2, 2, QImage.Format.Format_RGB888)
        img.fill(0)
        recorder.record(img)


class TestRecorderCloseFile:
    def test_positive_releases(self, recorder):
        with patch("app.core.Recorder.cv2.VideoWriter") as vw:
            mock_out = MagicMock()
            vw.return_value = mock_out
            recorder.open_file(4, 4, 10.0, "T", "avi")
            recorder.close_file()
            mock_out.release.assert_called_once()
            assert recorder.output is None
