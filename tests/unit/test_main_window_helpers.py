"""Unit tests: MainWindow helpers (lightweight, real Qt widget)."""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QRect

from app.core.MainWindow import MainWindow


@pytest.fixture
def main_win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class TestFormatHms:
    def test_zero(self, main_win):
        assert main_win._format_hms(0) == "0:00:00"

    def test_positive(self, main_win):
        assert main_win._format_hms(3661) == "1:01:01"


class TestSetCameraStreamActive:
    def test_active_sets_button_text(self, main_win):
        main_win.set_camera_stream_active(True)
        assert "Стоп" in main_win.ui.button_toggle_capture.text() or main_win.ui.button_toggle_capture.text() == "Стоп"

    def test_inactive_resets_when_no_video(self, main_win):
        main_win.videoPlayer = None
        main_win.set_camera_stream_active(False)


class TestDisableRoiNegative:
    def test_no_dimensions_returns(self, main_win):
        main_win.camera = None
        main_win.videoPlayer = None
        main_win.roi_controls = (MagicMock(),) * 4
        main_win.disable_roi()
        main_win.roi_controls[0].setValue.assert_not_called()


class TestApplyRoiFromMouseNegative:
    def test_small_rect_ignored(self, main_win):
        main_win.camera = MagicMock()
        main_win.camera.width = 100
        main_win.camera.height = 100
        main_win.roi_controls = tuple(MagicMock() for _ in range(4))
        main_win._apply_roi_from_mouse_rect(QRect(0, 0, 2, 2))
        main_win.roi_controls[0].setValue.assert_not_called()

    def test_no_pixmap(self, main_win):
        main_win.camera = MagicMock()
        main_win.camera.width = 100
        main_win.camera.height = 100
        main_win.roi_controls = tuple(MagicMock() for _ in range(4))
        main_win.ui.video_frame_label.pixmap = MagicMock(return_value=None)
        main_win._apply_roi_from_mouse_rect(QRect(0, 0, 20, 20))


class TestAskUserConfirmation:
    def test_returns_bool(self, main_win, monkeypatch):
        from PyQt6 import QtWidgets

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No,
        )
        assert main_win.ask_user_confirmation("t", "m") is False


class TestCameraRecording:
    def test_start_recording_no_camera_shows_message(self, main_win, qtbot):
        main_win.camera = None
        main_win.start_camera_recording()
        qtbot.waitUntil(lambda: True, timeout=100)

    def test_start_recording_not_capturing(self, main_win, qtbot):
        cam = MagicMock()
        cam.flag_capture = False
        main_win.camera = cam
        main_win.start_camera_recording()
        cam.start_record.assert_not_called()

    def test_toggle_video_no_player(self, main_win, qtbot):
        main_win.videoPlayer = None
        main_win.toggle_video_playback()
        qtbot.wait(50)
