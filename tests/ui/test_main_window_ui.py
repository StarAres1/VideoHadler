# tests/unit/test_main_window_ui.py
"""UI tests: MainWindow layout, signals, кнопки и rubber-band ROI (PyQt6 + pytest-qt)."""

from unittest.mock import MagicMock, patch, call

import pytest
from PyQt6.QtCore import QRect, QPoint, Qt, QEvent, QSize, QTimer
from PyQt6.QtGui import QColor, QImage, QPixmap, QMouseEvent
from PyQt6.QtWidgets import QWidget

from app.core.MainWindow import MainWindow
from app.core.Enums import ContrastImprovement, NoiseReduction
import numpy as np


@pytest.fixture
def main_win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def main_win_with_mocks(main_win):
    """MainWindow с замоканными зависимостями."""
    main_win.camera = MagicMock()
    main_win.camera.flag_capture = True
    main_win.camera.width = 640
    main_win.camera.height = 480
    main_win.videoPlayer = MagicMock()
    main_win.videoPlayer.is_loaded.return_value = True
    main_win.videoPlayer.width = 640
    main_win.videoPlayer.height = 480
    main_win.roi_controls = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    return main_win


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
        assert main_win.ui.roi_group.isEnabled() is False
        assert main_win.bt_disable_roi.isEnabled() is False

    def test_contrast_pipeline_radio_and_button_exist(self, main_win):
        assert hasattr(main_win, "radio_contrast_pipeline")
        assert hasattr(main_win, "button_contrast_pipeline_info")

    def test_zero_dce_radio_and_button_exist(self, main_win):
        assert hasattr(main_win, "radio_contrast_zero_dce")
        assert hasattr(main_win, "button_zero_dce_info")

    def test_frame_stats_button_exists(self, main_win):
        assert hasattr(main_win, "button_frame_stats")
        assert main_win.button_frame_stats.text() == "Статистика по кадру"


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

    def test_set_camera_stream_active_resets_roi_display_mode(self, main_win):
        main_win._reset_roi_display_mode_ui = MagicMock()
        main_win.set_camera_stream_active(True)
        main_win._reset_roi_display_mode_ui.assert_called_once()

    def test_set_camera_stream_active_stops_record_timer_when_false(self, main_win):
        main_win.stop_record_timer = MagicMock()
        main_win.set_camera_stream_active(False)
        main_win.stop_record_timer.assert_called_once()


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
        main_win._apply_roi_from_mouse_rect(QRect(62, 72, 68, 46))
        assert main_win.ui.spin_roi_w.value() >= 1
        assert main_win.ui.spin_roi_h.value() >= 1
        assert main_win.ui.spin_roi_x.value() >= 0
        assert main_win.ui.spin_roi_x.value() > 0
        assert main_win.ui.spin_roi_y.value() > 0


# ========== НОВЫЕ ТЕСТЫ ДЛЯ УВЕЛИЧЕНИЯ ПОКРЫТИЯ ==========

class TestMethodParamTitle:
    def test_method_param_title_returns_correct_titles(self, main_win):
        assert main_win._method_param_title(ContrastImprovement.CLAHE) == "Параметры CLAHE"
        assert main_win._method_param_title(ContrastImprovement.adjust_contrast) == "Параметры линейного преобразования"
        assert main_win._method_param_title(ContrastImprovement.gamma) == "Параметры гамма-коррекции"
        assert main_win._method_param_title(ContrastImprovement.sigmoid) == "Параметры сигмоидной коррекции"
        assert main_win._method_param_title(ContrastImprovement.autoGamma) == "Параметры автогаммы"
        assert main_win._method_param_title(ContrastImprovement.nn) == "Параметры нейросетевого метода"
        assert main_win._method_param_title(ContrastImprovement.zero_dce) == "Параметры Zero-DCE"
        # Default for unknown
        assert main_win._method_param_title(ContrastImprovement.HE) == "Параметры метода"


class TestConfigurePipelineMethod:
    def test_configure_pipeline_method_clahe(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_dialog_clahe") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.CLAHE)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)

    def test_configure_pipeline_method_adjust_contrast(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_dialog_adjust_contrast") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.adjust_contrast)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)

    def test_configure_pipeline_method_gamma(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_gamma_info") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.gamma)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)

    def test_configure_pipeline_method_sigmoid(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_sigmoid_info") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.sigmoid)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)

    def test_configure_pipeline_method_autoGamma(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_auto_gamma_info") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.autoGamma)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)

    def test_configure_pipeline_method_nn(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_nn_auto_info") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.nn)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)

    def test_configure_pipeline_method_zero_dce(self, main_win):
        with patch("app.core.ui_panels.processing_dialogs.show_zero_dce_info") as mock:
            main_win._configure_pipeline_method(ContrastImprovement.zero_dce)
            mock.assert_called_once_with(main_win, main_win._apply_to_active_sources)


class TestResolutionControls:
    def test_set_camera_resolution_options_populates_radio_buttons(self, main_win):
        resolutions = [(1920, 1080), (1280, 720), (640, 480)]
        selected = (1280, 720)
        callback = MagicMock()
        main_win.set_camera_resolution_options(resolutions, selected, callback)
        assert len(main_win.resolution_radio_buttons) == 3
        assert main_win.resolution_selected_callback == callback
        # Check that the selected radio is checked
        for rb in main_win.resolution_radio_buttons:
            if rb.text() == "1280 x 720":
                assert rb.isChecked() is True
            else:
                assert rb.isChecked() is False

    def test_set_camera_resolution_options_empty_resolutions(self, main_win):
        callback = MagicMock()
        main_win.set_camera_resolution_options([], None, callback)
        assert main_win.resolution_hint.text() == "Не удалось определить поддерживаемые разрешения"
        assert len(main_win.resolution_radio_buttons) == 0

    def test_on_resolution_radio_toggled_calls_callback_when_checked(self, main_win):
        callback = MagicMock()
        main_win.resolution_selected_callback = callback
        main_win._on_resolution_radio_toggled(True, 1920, 1080)
        callback.assert_called_once_with(1920, 1080)

    def test_on_resolution_radio_toggled_ignores_unchecked(self, main_win):
        callback = MagicMock()
        main_win.resolution_selected_callback = callback
        main_win._on_resolution_radio_toggled(False, 1920, 1080)
        callback.assert_not_called()


class TestActiveNativeFrameSize:
    def test_active_native_frame_size_from_camera(self, main_win_with_mocks):
        main_win_with_mocks.camera.width = 800
        main_win_with_mocks.camera.height = 600
        w, h = main_win_with_mocks.active_native_frame_size()
        assert w == 800
        assert h == 600

    def test_active_native_frame_size_from_video_player(self, main_win_with_mocks):
        main_win_with_mocks.camera = None
        main_win_with_mocks.videoPlayer.width = 1024
        main_win_with_mocks.videoPlayer.height = 768
        w, h = main_win_with_mocks.active_native_frame_size()
        assert w == 1024
        assert h == 768

    def test_active_native_frame_size_returns_zero_when_no_source(self, main_win):
        main_win.camera = None
        main_win.videoPlayer = MagicMock()
        main_win.videoPlayer.is_loaded.return_value = False
        w, h = main_win.active_native_frame_size()
        assert w == 0
        assert h == 0


class TestRoiOverlay:
    def test_roi_overlay_parent_widget_returns_parent(self, main_win):
        parent = main_win._roi_overlay_parent_widget()
        assert parent is not None

    def test_init_roi_overlay_frames_creates_frames(self, main_win):
        assert hasattr(main_win, "_roi_box_frame")
        assert hasattr(main_win, "_roi_drag_frame")
        assert main_win._roi_box_frame.parent() is not None
        assert main_win._roi_drag_frame.parent() is not None

    def test_refresh_roi_overlay_hides_when_content_display_active(self, main_win):
        main_win._roi_content_display_active = True
        main_win._roi_box_frame.hide = MagicMock()
        main_win.refresh_roi_overlay()
        main_win._roi_box_frame.hide.assert_called_once()

    def test_refresh_roi_overlay_hides_when_no_dimensions(self, main_win):
        main_win._roi_content_display_active = False
        main_win._frame_pixel_size = MagicMock(return_value=(0, 0))
        main_win._roi_box_frame.hide = MagicMock()
        main_win.refresh_roi_overlay()
        main_win._roi_box_frame.hide.assert_called_once()

    def test_refresh_roi_overlay_hides_when_full_frame_roi(self, main_win):
        main_win._roi_content_display_active = False
        main_win._frame_pixel_size = MagicMock(return_value=(640, 480))
        main_win.roi_controls = (
            MagicMock(value=MagicMock(return_value=0)),
            MagicMock(value=MagicMock(return_value=0)),
            MagicMock(value=MagicMock(return_value=640)),
            MagicMock(value=MagicMock(return_value=480)),
        )
        main_win._video_pixmap_rect_in_label = MagicMock(return_value=QRect(0, 0, 640, 480))
        main_win._roi_box_frame.hide = MagicMock()
        main_win.refresh_roi_overlay()
        main_win._roi_box_frame.hide.assert_called_once()

    def test_clamp_rect_to_video_area_returns_intersection(self, main_win):
        main_win._video_pixmap_rect_in_label = MagicMock(return_value=QRect(10, 10, 100, 100))
        rect = QRect(5, 5, 50, 50)
        result = main_win._clamp_rect_to_video_area(rect)
        assert result.x() == 10
        assert result.y() == 10

    def test_clamp_rect_to_video_area_returns_original_if_no_area(self, main_win):
        main_win._video_pixmap_rect_in_label = MagicMock(return_value=None)
        rect = QRect(5, 5, 50, 50)
        result = main_win._clamp_rect_to_video_area(rect)
        assert result.x() == 5


class TestEventFilter:
    def test_event_filter_resize_triggers_refresh_roi_overlay(self, main_win):
        main_win.refresh_roi_overlay = MagicMock()
        event = QEvent(QEvent.Type.Resize)
        result = main_win.eventFilter(main_win.ui.video_frame_label, event)
        main_win.refresh_roi_overlay.assert_called_once()
        assert result is False

class TestRecordTimer:
    def test_format_hms(self, main_win):
        assert main_win._format_hms(0) == "0:00:00"
        assert main_win._format_hms(61) == "0:01:01"
        assert main_win._format_hms(3661) == "1:01:01"

    def test_update_record_timer_increments_and_updates_label(self, main_win):
        main_win.record_elapsed_sec = 10
        main_win.ui.label_record_time_value.setText = MagicMock()
        main_win._update_record_timer()
        assert main_win.record_elapsed_sec == 11
        main_win.ui.label_record_time_value.setText.assert_called_once_with("0:00:11")

    def test_stop_record_timer_stops_and_resets_label(self, main_win):
        main_win.record_timer.stop = MagicMock()
        main_win.stop_record_timer()
        main_win.record_timer.stop.assert_called_once()
        assert main_win.ui.label_record_time_value.text() == "0:00:00"


class TestCameraRecording:

    def test_start_camera_recording_not_capturing(self, main_win):
        cam = MagicMock()
        cam.flag_capture = False
        main_win.camera = cam
        main_win.start_camera_recording()
        cam.start_record.assert_not_called()

    def test_start_camera_recording_success(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        main_win.camera = cam
        main_win.start_record_timer = MagicMock()
        main_win.start_camera_recording()
        cam.set_record_format.assert_called_once_with("avi")
        cam.start_record.assert_called_once_with("avi")
        main_win.start_record_timer.assert_called_once()
        assert main_win.ui.button_toggle_recording.text() == "Завершить запись"

    def test_stop_camera_recording(self, main_win):
        cam = MagicMock()
        main_win.camera = cam
        main_win.stop_record_timer = MagicMock()
        main_win.stop_camera_recording()
        cam.stop_record.assert_called_once()
        main_win.stop_record_timer.assert_called_once()
        assert main_win.ui.button_toggle_recording.text() == "Запись в файл"

    def test_toggle_camera_recording_when_not_recording(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.flag_record = False
        main_win.camera = cam
        main_win.start_camera_recording = MagicMock()
        main_win.toggle_camera_recording()
        main_win.start_camera_recording.assert_called_once()

    def test_toggle_camera_recording_when_recording(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.flag_record = True
        main_win.camera = cam
        main_win.stop_camera_recording = MagicMock()
        main_win.toggle_camera_recording()
        main_win.stop_camera_recording.assert_called_once()


class TestVideoPlayback:

    def test_toggle_video_playback_with_player(self, main_win):
        vp = MagicMock()
        vp.is_loaded.return_value = True
        main_win.videoPlayer = vp
        main_win.toggle_video_playback()
        vp.toggle_play_pause.assert_called_once()

    def test_video_seek_backward(self, main_win):
        vp = MagicMock()
        main_win.videoPlayer = vp
        main_win.video_seek_backward()
        vp.seek_seconds.assert_called_once_with(-10)

    def test_video_seek_forward(self, main_win):
        vp = MagicMock()
        main_win.videoPlayer = vp
        main_win.video_seek_forward()
        vp.seek_seconds.assert_called_once_with(10)

    def test_set_video_position(self, main_win):
        vp = MagicMock()
        main_win.videoPlayer = vp
        main_win.set_video_position(75)
        vp.set_position_percent.assert_called_once_with(75)


class TestFrameStatistics:
    def test_get_current_frame_pair_for_statistics_from_camera(self, main_win_with_mocks):
        main_win_with_mocks.camera.last_preview_before_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        main_win_with_mocks.camera.last_preview_after_rgb = np.ones((480, 640, 3), dtype=np.uint8)
        before, after, source = main_win_with_mocks._get_current_frame_pair_for_statistics()
        assert source == "camera"
        assert before is not None
        assert after is not None

    def test_get_current_frame_pair_for_statistics_from_file(self, main_win_with_mocks):
        main_win_with_mocks.camera = None
        main_win_with_mocks.videoPlayer.get_frame_pair_for_statistics.return_value = (np.zeros((10,10,3)), np.ones((10,10,3)))
        before, after, source = main_win_with_mocks._get_current_frame_pair_for_statistics()
        assert source == "file"
        assert before is not None
        assert after is not None

    def test_get_current_frame_pair_for_statistics_returns_none_when_no_data(self, main_win):
        main_win.camera = MagicMock()
        main_win.camera.flag_capture = True
        main_win.camera.last_preview_before_rgb = None
        before, after, source = main_win._get_current_frame_pair_for_statistics()
        assert before is None
        assert after is None
        assert source is None

    def test_pause_preview_for_stats_camera(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.preview_paused = False
        main_win.camera = cam
        main_win._pause_preview_for_stats("camera")
        assert cam.preview_paused is True
        assert main_win._frame_stats_camera_was_paused is False

    def test_pause_preview_for_stats_camera_already_paused(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.preview_paused = True
        main_win.camera = cam
        main_win._pause_preview_for_stats("camera")
        assert main_win._frame_stats_camera_was_paused is True
        # preview_paused remains True (was already paused)

    def test_pause_preview_for_stats_file(self, main_win):
        vp = MagicMock()
        vp.is_loaded.return_value = True
        vp.is_playing.return_value = True
        main_win.videoPlayer = vp
        main_win._pause_preview_for_stats("file")
        assert main_win._frame_stats_resume_file is True
        vp.pause.assert_called_once()

    def test_pause_preview_for_stats_file_not_playing(self, main_win):
        vp = MagicMock()
        vp.is_loaded.return_value = True
        vp.is_playing.return_value = False
        main_win.videoPlayer = vp
        main_win._pause_preview_for_stats("file")
        assert main_win._frame_stats_resume_file is False
        vp.pause.assert_not_called()

    def test_resume_preview_after_stats_camera(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.preview_paused = True
        main_win.camera = cam
        main_win._frame_stats_camera_was_paused = True
        main_win._resume_preview_after_stats()
        assert cam.preview_paused is True  # restored to original state

    def test_resume_preview_after_stats_camera_was_not_paused(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.preview_paused = True
        main_win.camera = cam
        main_win._frame_stats_camera_was_paused = False
        main_win._resume_preview_after_stats()
        assert cam.preview_paused is False

    def test_resume_preview_after_stats_file(self, main_win):
        vp = MagicMock()
        vp.is_loaded.return_value = True
        main_win.videoPlayer = vp
        main_win._frame_stats_resume_file = True
        main_win._resume_preview_after_stats()
        vp.resume.assert_called_once()
        assert main_win._frame_stats_resume_file is False

    def test_contrast_method_display_name(self, main_win):
        assert main_win._contrast_method_display_name(ContrastImprovement.NotImprove) == "Без улучшения"
        assert main_win._contrast_method_display_name(ContrastImprovement.CLAHE) == "CLAHE"
        assert main_win._contrast_method_display_name(ContrastImprovement.adjust_contrast) == "Линейное преобразование"
        assert main_win._contrast_method_display_name(ContrastImprovement.HE) == "Эквализация гистограммы (HE)"
        assert main_win._contrast_method_display_name(ContrastImprovement.gamma) == "Гамма-коррекция"
        assert main_win._contrast_method_display_name(ContrastImprovement.autoGamma) == "Автогамма"
        assert main_win._contrast_method_display_name(ContrastImprovement.sigmoid) == "Сигмоидная коррекция"
        assert main_win._contrast_method_display_name(ContrastImprovement.nn) == "Автоподбор нейросетью"
        assert main_win._contrast_method_display_name(ContrastImprovement.pipeline) == "Цепочка методов"
        assert main_win._contrast_method_display_name(ContrastImprovement.zero_dce) == "Zero-DCE"

    def test_contrast_method_with_params_clahe(self, main_win):
        processor = MagicMock()
        processor.config.clip_limit = 2.5
        processor.config.tile_grid_size = 8
        result = main_win._contrast_method_with_params(ContrastImprovement.CLAHE, processor)
        assert "CLAHE" in result
        assert "clipLimit=2.50" in result
        assert "tileGrid=8" in result

    def test_contrast_method_with_params_adjust_contrast(self, main_win):
        processor = MagicMock()
        processor.config.alpha = 1.2
        processor.config.beta = 15
        result = main_win._contrast_method_with_params(ContrastImprovement.adjust_contrast, processor)
        assert "Линейное преобразование" in result
        assert "alpha=1.20" in result
        assert "beta=15" in result

    def test_contrast_method_with_params_gamma(self, main_win):
        processor = MagicMock()
        processor.config.gamma = 2.2
        result = main_win._contrast_method_with_params(ContrastImprovement.gamma, processor)
        assert "Гамма-коррекция" in result
        assert "gamma=2.20" in result

    def test_contrast_method_with_params_nn(self, main_win):
        processor = MagicMock()
        processor.config.nn_skip_frames = 5
        processor._nn_last_label = "gamma_1.5"
        result = main_win._contrast_method_with_params(ContrastImprovement.nn, processor)
        assert "Автоподбор нейросетью" in result
        assert "skip_frames=5" in result
        assert "выбранный метод=gamma_1.5" in result

    def test_build_applied_contrast_text_no_source(self, main_win):
        main_win.camera = None
        main_win.videoPlayer = None
        result = main_win._build_applied_contrast_text("camera")
        assert "Источник не определён" in result

    def test_build_applied_contrast_text_pipeline_empty(self, main_win):
        cam = MagicMock()
        cam.method_for_contrast = ContrastImprovement.pipeline
        cam.contrast_pipeline = []
        proc = MagicMock()
        proc.config = MagicMock()
        cam.video_handler = MagicMock()
        cam.video_handler.processor = proc
        main_win.camera = cam
        result = main_win._build_applied_contrast_text("camera")
        assert "Цепочка пустая" in result

class TestApplyToActiveSources:
    def test_apply_to_active_sources_calls_camera_method_with_value(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        main_win.camera = cam
        vp = MagicMock()
        vp.is_loaded.return_value = False
        main_win.videoPlayer = vp
        main_win._apply_to_active_sources("set_gamma_value", 1.5)
        cam.set_gamma_value.assert_called_once_with(1.5)

    def test_apply_to_active_sources_calls_camera_method_without_value(self, main_win):
        cam = MagicMock()
        cam.flag_capture = True
        cam.refresh_current_frame = MagicMock()
        main_win.camera = cam
        main_win._apply_to_active_sources("refresh_current_frame", None)
        cam.refresh_current_frame.assert_called_once()

    def test_apply_to_active_sources_calls_video_player_method(self, main_win):
        cam = MagicMock()
        cam.flag_capture = False
        main_win.camera = cam
        vp = MagicMock()
        vp.is_loaded.return_value = True
        main_win.videoPlayer = vp
        main_win._apply_to_active_sources("set_gamma_value", 2.0)
        vp.set_gamma_value.assert_called_once_with(2.0)


class TestSetRoiControlsAndCallback:
    def test_set_roi_controls_stores_references(self, main_win):
        spins = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        main_win.set_roi_controls(*spins)
        assert main_win.roi_controls == spins

    def test_set_roi_change_callback_stores_callback(self, main_win):
        cb = MagicMock()
        main_win.set_roi_change_callback(cb)
        assert main_win.roi_change_callback == cb

    def test_roi_change_callback_invoked_on_disable_roi(self, main_win_with_mocks):
        main_win_with_mocks.roi_change_callback = MagicMock()
        main_win_with_mocks.disable_roi()
        main_win_with_mocks.roi_change_callback.assert_called_once()


class TestResetRoiDisplayModeUi:
    def test_reset_roi_display_mode_ui_unchecks_button_and_calls_callback(self, main_win):
        main_win.ui.button_toggle_roi_display = MagicMock()
        main_win.ui.button_toggle_roi_display.blockSignals = MagicMock()
        main_win.ui.button_toggle_roi_display.setChecked = MagicMock()
        main_win.roi_display_applied_callback = MagicMock()
        main_win.set_roi_content_display_active = MagicMock()
        main_win._reset_roi_display_mode_ui()
        main_win.ui.button_toggle_roi_display.setChecked.assert_called_once_with(False)
        main_win.roi_display_applied_callback.assert_called_once_with(False)
        main_win.set_roi_content_display_active.assert_called_once_with(False)


class TestRefreshRoiOverlayEdgeCases:
    def test_refresh_roi_overlay_no_roi_box_frame(self, main_win):
        del main_win._roi_box_frame
        main_win.refresh_roi_overlay()  # should not crash

    def test_refresh_roi_overlay_no_roi_controls(self, main_win):
        main_win._roi_box_frame = MagicMock()
        main_win.roi_controls = None
        main_win.refresh_roi_overlay()
        main_win._roi_box_frame.hide.assert_called_once()

    def test_refresh_roi_overlay_invalid_video_area(self, main_win):
        main_win._roi_content_display_active = False
        main_win._frame_pixel_size = MagicMock(return_value=(640, 480))
        main_win.roi_controls = (MagicMock(value=MagicMock(return_value=0)),) * 4
        main_win._video_pixmap_rect_in_label = MagicMock(return_value=None)
        main_win._roi_box_frame.hide = MagicMock()
        main_win.refresh_roi_overlay()
        main_win._roi_box_frame.hide.assert_called_once()


class TestVideoPixmapRectInLabel:
    def test_video_pixmap_rect_in_label_no_pixmap(self, main_win):
        main_win.ui.video_frame_label.pixmap = MagicMock(return_value=None)
        assert main_win._video_pixmap_rect_in_label() is None

    def test_video_pixmap_rect_in_label_with_pixmap(self, main_win):
        pixmap = QPixmap(100, 80)
        main_win.ui.video_frame_label.setPixmap(pixmap)
        main_win.ui.video_frame_label.setFixedSize(200, 160)
        rect = main_win._video_pixmap_rect_in_label()
        assert rect.width() == 100
        assert rect.height() == 80
        assert rect.x() == (200 - 100) // 2
        assert rect.y() == (160 - 80) // 2


class TestMapRoiRectLabelToOverlayParent:
    def test_map_roi_rect_label_to_overlay_parent_same_parent(self, main_win):
        # When parent is the label itself
        main_win._roi_overlay_parent_widget = MagicMock(return_value=main_win.ui.video_frame_label)
        rect = QRect(10, 20, 30, 40)
        result = main_win._map_roi_rect_label_to_overlay_parent(rect)
        assert result == rect

class TestFramePixelSize:
    def test_frame_pixel_size_returns_active_native_frame_size(self, main_win_with_mocks):
        size = main_win_with_mocks._frame_pixel_size()
        assert size == (640, 480)


class TestUpdateRoiToggleButtonText:
    def test_update_roi_toggle_button_text_checked(self, main_win):
        main_win.ui.button_toggle_roi_display = MagicMock()
        main_win.ui.button_toggle_roi_display.isChecked.return_value = True
        main_win._update_roi_toggle_button_text()
        assert main_win.ui.button_toggle_roi_display.setText.call_args[0][0] == "Показать весь кадр"

    def test_update_roi_toggle_button_text_unchecked(self, main_win):
        main_win.ui.button_toggle_roi_display = MagicMock()
        main_win.ui.button_toggle_roi_display.isChecked.return_value = False
        main_win._update_roi_toggle_button_text()
        assert main_win.ui.button_toggle_roi_display.setText.call_args[0][0] == "Показать область ROI на видео"


class TestSetRoiContentDisplayActive:
    def test_set_roi_content_display_active_hides_drag_frame(self, main_win):
        main_win._roi_drag_frame.hide = MagicMock()
        main_win._update_roi_toggle_button_text = MagicMock()
        main_win.refresh_roi_overlay = MagicMock()
        main_win.set_roi_content_display_active(True)
        assert main_win._roi_content_display_active is True
        main_win._roi_drag_frame.hide.assert_called_once()
        main_win._update_roi_toggle_button_text.assert_called_once()
        main_win.refresh_roi_overlay.assert_called_once()


class TestSetProcessingBlocksEnabled:
    def test_set_processing_blocks_enabled_false(self, main_win):
        main_win.ui.recording_group.setEnabled = MagicMock()
        main_win.ui.contrast_group.setEnabled = MagicMock()
        main_win.ui.noise_reduction_group.setEnabled = MagicMock()
        main_win.ui.roi_group.setEnabled = MagicMock()
        main_win._set_processing_blocks_enabled(False)
        main_win.ui.recording_group.setEnabled.assert_called_with(False)
        main_win.ui.contrast_group.setEnabled.assert_called_with(False)
        main_win.ui.noise_reduction_group.setEnabled.assert_called_with(False)
        main_win.ui.roi_group.setEnabled.assert_called_with(False)