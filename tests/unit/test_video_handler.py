"""Unit tests: VideoHandler signal API (no camera loop)."""

from unittest.mock import MagicMock

from app.core.VideoHandler import VideoHandler


def test_video_handler_has_expected_signals():
    cam = MagicMock()
    vh = VideoHandler(cam)
    assert hasattr(vh, "paint_frame")
    assert hasattr(vh, "open_file")
    assert hasattr(vh, "record_frame")
    assert hasattr(vh, "close_file")
    assert hasattr(vh, "show_fps")
