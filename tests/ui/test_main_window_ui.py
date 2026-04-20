"""UI tests: MainWindow layout, signals, кнопки и rubber-band ROI (PyQt6 + pytest-qt)."""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QRect, QPoint, Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from app.core.MainWindow import MainWindow


@pytest.fixture
def main_win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class TestMainWindowInitialUi:
    def test_video_frame_exists_and_has_minimum_size(self, main_win):
        assert main_win.ui.video_frame_label.minimumWidth() >= 600
        assert main_win.ui.video_frame_label.minimumHeight() >= 400

    def test_file_playback_group_hidden_initially(self, main_win):
        assert main_win.ui.playback_group.isVisible() is False

    def test_processing_blocks_disabled_until_stream(self, main_win):
        assert main_win.ui.recording_group.isEnabled() is False
        assert main_win.ui.contrast_group.isEnabled() is False
        assert main_win.ui.noise_reduction_group.isEnabled() is False
        assert main_win.ui.roi_group.isEnabled() is False

    def test_record_format_combo_items(self, main_win):
        assert main_win.ui.combo_record_format.count() == 2
        assert main_win.ui.combo_record_format.itemText(0) == "avi"
        assert main_win.ui.combo_record_format.itemText(1) == "mp4"
        assert main_win.ui.combo_record_format.currentText() == "avi"

    def test_capture_button_initial_label(self, main_win):
        assert main_win.ui.button_toggle_capture.text() == "Старт"

    def test_disable_roi_button_present(self, main_win):
        assert main_win.bt_disable_roi.text() == "Отключить ROI"
        # Пока roi_group выключен, дочерняя кнопка тоже неактивна — это ожидаемо.
        assert main_win.ui.roi_group.isEnabled() is False
        assert main_win.bt_disable_roi.isEnabled() is False


class TestMainWindowStreamAndProcessingUi:
    def test_active_stream_enables_processing_blocks(self, main_win):
        main_win.set_camera_stream_active(True)
        assert main_win.ui.recording_group.isEnabled() is True
        assert main_win.ui.contrast_group.isEnabled() is True
        assert main_win.ui.noise_reduction_group.isEnabled() is True
        assert main_win.ui.roi_group.isEnabled() is True
        assert main_win.ui.button_toggle_capture.text() == "Стоп"

    def test_inactive_stream_disables_when_no_video(self, main_win):
        main_win.set_camera_stream_active(True)
        main_win.videoPlayer = None
        main_win.set_camera_stream_active(False)
        assert main_win.ui.recording_group.isEnabled() is False


class TestMainWindowVideoSignalsUi:
    def test_position_slider_updates_from_signal(self, main_win):
        main_win._on_video_position_changed(37)
        assert main_win.ui.slider_playback_position.value() == 37

    def test_playback_state_updates_play_button_text(self, main_win):
        main_win.videoPlayer = MagicMock()
        main_win.videoPlayer.is_loaded = MagicMock(return_value=True)
        main_win._on_playback_state_changed(True)
        assert main_win.ui.button_toggle_playback.text() == "Пауза"
        main_win._on_playback_state_changed(False)
        assert main_win.ui.button_toggle_playback.text() == "Старт"

    def test_file_opened_updates_filename_label_and_shows_group(self, main_win, qtbot, tmp_path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"")
        main_win.show()
        qtbot.waitExposed(main_win)
        main_win._on_file_opened(str(path))
        assert main_win.ui.label_selected_file_name.text() == "clip.mp4"
        assert main_win.ui.playback_group.isVisible() is True
        assert main_win.ui.recording_group.isEnabled() is True

    def test_show_file_fps_sets_status_message(self, main_win, qtbot):
        main_win.show_file_fps(30.5)
        qtbot.wait(0)
        msg = main_win.statusBar().currentMessage()
        assert "FPS" in msg
        assert "30.50" in msg or "30.5" in msg


class TestMainWindowScreenshotAndPlaybackUi:
    def test_screenshot_without_pixmap_sets_status(self, main_win, qtbot):
        main_win.ui.video_frame_label.clear()
        main_win.make_video_screenshot()
        qtbot.wait(0)
        assert "скриншот" in main_win.statusBar().currentMessage().lower()

    def test_toggle_playback_without_file_sets_status(self, main_win, qtbot):
        main_win.videoPlayer = MagicMock()
        main_win.videoPlayer.is_loaded.return_value = False
        main_win.toggle_video_playback()
        qtbot.wait(0)
        assert len(main_win.statusBar().currentMessage()) > 0


class TestDisableRoiButtonClick:
    def test_click_resets_spin_boxes(self, main_win, qtbot):
        cam = MagicMock()
        cam.width = 320
        cam.height = 240
        main_win.camera = cam
        main_win.set_roi_controls(
            main_win.ui.spin_roi_x,
            main_win.ui.spin_roi_y,
            main_win.ui.spin_roi_w,
            main_win.ui.spin_roi_h,
        )
        main_win._set_processing_blocks_enabled(True)
        for sb in (
            main_win.ui.spin_roi_x,
            main_win.ui.spin_roi_y,
            main_win.ui.spin_roi_w,
            main_win.ui.spin_roi_h,
        ):
            sb.setMaximum(10000)
        main_win.ui.spin_roi_x.setValue(10)
        main_win.ui.spin_roi_y.setValue(20)
        main_win.ui.spin_roi_w.setValue(100)
        main_win.ui.spin_roi_h.setValue(80)
        qtbot.mouseClick(main_win.bt_disable_roi, Qt.MouseButton.LeftButton)
        assert main_win.ui.spin_roi_x.value() == 0
        assert main_win.ui.spin_roi_y.value() == 0
        assert main_win.ui.spin_roi_w.value() == 320
        assert main_win.ui.spin_roi_h.value() == 240


class TestRoiSelectionUpdatesSpinBoxes:
    """Тот же код, что после отпускания мыши на video_frame_label (см. eventFilter → _apply_roi_from_mouse_rect)."""

    def test_apply_roi_rect_updates_width_height_spins(self, main_win, qtbot):
        main_win.camera = MagicMock()
        main_win.camera.width = 100
        main_win.camera.height = 100
        main_win.set_roi_controls(
            main_win.ui.spin_roi_x,
            main_win.ui.spin_roi_y,
            main_win.ui.spin_roi_w,
            main_win.ui.spin_roi_h,
        )
        for sb in (
            main_win.ui.spin_roi_x,
            main_win.ui.spin_roi_y,
            main_win.ui.spin_roi_w,
            main_win.ui.spin_roi_h,
        ):
            sb.setMaximum(10000)
        main_win._set_processing_blocks_enabled(True)
        img = QImage(80, 60, QImage.Format.Format_RGB888)
        img.fill(QColor(40, 50, 60))
        main_win.ui.video_frame_label.setPixmap(QPixmap.fromImage(img))
        main_win.ui.video_frame_label.setFixedSize(200, 200)
        main_win.show()
        qtbot.waitExposed(main_win)
        qtbot.wait(0)
        # Прямоугольник в координатах label_record_format 200×200: pixmap 80×60 по центру → offset (60,70).
        main_win._apply_roi_from_mouse_rect(QRect(62, 72, 68, 46))
        assert main_win.ui.spin_roi_w.value() >= 1
        assert main_win.ui.spin_roi_h.value() >= 1
        assert main_win.ui.spin_roi_x.value() >= 0
        # Regression: first ROI placement must not snap to zero offsets.
        assert main_win.ui.spin_roi_x.value() > 0
        assert main_win.ui.spin_roi_y.value() > 0


class TestClaheDialogOpens:
    def test_dialog_visible_when_source_ready(self, main_win, qtbot):
        main_win.videoPlayer = MagicMock()
        main_win.videoPlayer.is_loaded.return_value = True
        main_win.show_dialog_CLAHE()
        qtbot.waitUntil(lambda: getattr(main_win, "dialog_clahe", None) is not None, timeout=2000)
        assert main_win.dialog_clahe.isVisible()
