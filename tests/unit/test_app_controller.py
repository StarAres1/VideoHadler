"""Unit tests: AppController with mocked MainWindow and subsystems."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.AppController import AppController
from app.core.Enums import ContrastImprovement


def _ui_names():
    return [
        "video_frame_label",
        "combo_cameras",
        "file_tree_view",
        "slider_roi_x",
        "spin_roi_x",
        "slider_roi_y",
        "spin_roi_y",
        "slider_roi_w",
        "spin_roi_w",
        "slider_roi_h",
        "spin_roi_h",
        "playback_group",
        "label_playback_time",
        "button_refresh_cameras",
        "button_toggle_capture",
        "button_toggle_recording",
        "combo_record_format",
        "radio_contrast_clahe",
        "radio_contrast_none",
        "radio_contrast_adjust",
        "radio_contrast_he",
        "radio_contrast_gamma",
        "radio_contrast_sigmoid",
        "radio_contrast_nn",
        "radio_contrast_auto_gamma",
        "radio_noise_none",
        "radio_noise_median",
        "radio_noise_fast_gaussian",
        "check_monochrome",
        "button_clahe_info",
        "button_adjust_info",
        "button_gamma_info",
        "button_sigmoid_info",
        "button_nn_auto_info",
        "button_auto_gamma_info",
        "button_noise_median_info",
        "button_noise_fast_gaussian_info",
        "button_toggle_playback",
        "button_seek_backward",
        "button_seek_forward",
        "button_take_screenshot",
        "slider_playback_position",
        "button_toggle_roi_display",
    ]


def make_main_mock(camera=None):
    main = MagicMock()
    ui = MagicMock()
    for name in _ui_names():
        m = MagicMock()
        m.value.return_value = 0
        setattr(ui, name, m)
    ui.combo_cameras.currentIndex.return_value = 0
    main.ui = ui
    main.camera = camera
    main.videoPlayer = None
    main.bind_video_player = MagicMock()
    main.set_roi_controls = MagicMock()
    main.set_roi_change_callback = MagicMock()
    main.refresh_roi_overlay = MagicMock()
    main.roi_display_applied_callback = None
    main.ask_user_confirmation = MagicMock(return_value=True)
    main.stop_camera_recording = MagicMock()
    main.set_camera_stream_active = MagicMock()
    main.toggle_camera_recording = MagicMock()
    main.ensure_nn_model_loaded_async = MagicMock()
    main.set_camera_resolution_options = MagicMock()
    main.set_camera_resolution_loading = MagicMock()
    main.show_dialog_CLAHE = MagicMock()
    main.show_dialog_adjustContrast = MagicMock()
    main.show_gamma_info = MagicMock()
    main.show_sigmoid_info = MagicMock()
    main.show_nn_auto_info = MagicMock()
    main.show_auto_gamma_info = MagicMock()
    main.show_noise_median_info = MagicMock()
    main.show_noise_nlm_info = MagicMock()
    main.toggle_video_playback = MagicMock()
    main.video_seek_backward = MagicMock()
    main.video_seek_forward = MagicMock()
    main.make_video_screenshot = MagicMock()
    main.set_video_position = MagicMock()
    main.active_native_frame_size = MagicMock(return_value=(640, 480))
    main.tree = MagicMock()
    main.tree.video_selected = MagicMock()
    main.tree.video_selected.connect = MagicMock()
    return main


@pytest.fixture
def patched_deps():
    with patch("app.core.AppController.CameraManager") as CM, patch(
        "app.core.AppController.FileBrowser"
    ) as FB, patch("app.core.AppController.VideoPlayer") as VP:
        mgr = CM.return_value
        mgr.find_cameras = MagicMock()
        mgr.current_camera.return_value = None
        vp = VP.return_value
        vp.is_loaded.return_value = False
        yield CM, FB, VP, vp


def test_init_sets_up_sources(patched_deps):
    CM, FB, VP, vp = patched_deps
    main = make_main_mock()
    ac = AppController(main)
    mgr = CM.return_value
    mgr.find_cameras.assert_called()
    main.bind_video_player.assert_called_once_with(vp)
    assert ac.current_camera_index[0] == 0


def test_on_camera_changed_negative_index_ignored(patched_deps):
    main = make_main_mock()
    ac = AppController(main)
    main.ui.combo_cameras.blockSignals.reset_mock()
    ac._on_camera_changed(-1)
    main.ui.combo_cameras.blockSignals.assert_not_called()


def test_toggle_capture_no_camera_returns(patched_deps):
    main = make_main_mock(camera=None)
    ac = AppController(main)
    ac._toggle_capture()
    main.set_camera_stream_active.assert_not_called()


def test_apply_to_active_sources_no_targets(patched_deps):
    main = make_main_mock(camera=None)
    ac = AppController(main)
    main.videoPlayer = MagicMock()
    main.videoPlayer.is_loaded.return_value = False
    ac._apply_to_active_sources("set_gamma_value", 1.5)


def test_apply_to_active_sources_calls_camera(patched_deps):
    CM, FB, VP, vp = patched_deps
    cam = MagicMock()
    cam.flag_capture = True
    cam.width = 640
    cam.height = 480
    cam.set_gamma_value = MagicMock()
    cam.set_roi_x = MagicMock()
    cam.set_roi_y = MagicMock()
    cam.set_roi_width = MagicMock()
    cam.set_roi_height = MagicMock()
    CM.return_value.current_camera.return_value = cam
    main = make_main_mock(camera=None)
    ac = AppController(main)
    assert main.camera is cam
    main.videoPlayer = MagicMock()
    main.videoPlayer.is_loaded.return_value = False
    ac._apply_to_active_sources("set_gamma_value", 2.0)
    cam.set_gamma_value.assert_called_once_with(2.0)


def test_stop_file_playback_hides_group(patched_deps):
    main = make_main_mock()
    ac = AppController(main)
    vp = MagicMock()
    vp.is_loaded.return_value = True
    main.videoPlayer = vp
    main.camera = None
    ac._stop_file_playback()
    vp.stop.assert_called_once()
    main.ui.playback_group.setVisible.assert_called_with(False)


def test_on_camera_changed_user_declines_stops_switch(patched_deps):
    main = make_main_mock()
    ac = AppController(main)
    main.videoPlayer = MagicMock()
    main.videoPlayer.is_loaded.return_value = True
    main.ask_user_confirmation.return_value = False
    ac.current_camera_index[0] = 0
    main.ui.combo_cameras.currentIndex.return_value = 1
    ac._on_camera_changed(2)
    main.ui.combo_cameras.setCurrentIndex.assert_called_with(0)
    main.ui.combo_cameras.blockSignals.assert_called()


def test_update_roi_limits_with_video_player_only(patched_deps):
    main = make_main_mock(camera=None)
    ac = AppController(main)
    vp = MagicMock()
    vp.is_loaded.return_value = True
    vp.width = 320
    vp.height = 240
    main.videoPlayer = vp
    main.active_native_frame_size = MagicMock(return_value=(320, 240))
    main.ui.spin_roi_x.value.return_value = 0
    main.ui.spin_roi_y.value.return_value = 0
    main.ui.spin_roi_w.value.return_value = 100
    main.ui.spin_roi_h.value.return_value = 100
    ac._update_roi_limits()
    main.ui.slider_roi_w.setMaximum.assert_called()


def test_update_roi_limits_prioritizes_roi_size_and_clamps_offsets(patched_deps):
    main = make_main_mock(camera=None)
    ac = AppController(main)
    vp = MagicMock()
    vp.is_loaded.return_value = True
    main.videoPlayer = vp
    main.active_native_frame_size = MagicMock(return_value=(320, 240))
    main.ui.spin_roi_w.value.return_value = 300
    main.ui.spin_roi_h.value.return_value = 220
    main.ui.spin_roi_x.value.return_value = 50
    main.ui.spin_roi_y.value.return_value = 30

    ac._update_roi_limits()

    # Offsets shrink according to current ROI size.
    main.ui.spin_roi_x.setMaximum.assert_called_with(20)
    main.ui.spin_roi_y.setMaximum.assert_called_with(20)
    # ROI size limits stay tied to frame size, not offset.
    main.ui.spin_roi_w.setMaximum.assert_called_with(320)
    main.ui.spin_roi_h.setMaximum.assert_called_with(240)


def test_start_resolution_probe_queues_when_thread_running(patched_deps):
    main = make_main_mock(camera=None)
    ac = AppController(main)
    running_thread = MagicMock()
    running_thread.isRunning.return_value = True
    ac._resolution_probe_thread = running_thread
    prev_req_id = ac._resolution_probe_request_id

    ac._start_resolution_probe(2)

    assert ac._pending_probe_camera_index == 2
    assert ac._resolution_probe_request_id == prev_req_id


def test_operator_contrast_updates_camera_and_player_state(patched_deps):
    main = make_main_mock()
    ac = AppController(main)
    cam = MagicMock()
    cam.video_handler = MagicMock()
    cam.video_handler.processor = MagicMock()
    cam.video_handler.processor.config = MagicMock()
    vp = MagicMock()
    vp.processor = MagicMock()
    vp.processor.config = MagicMock()
    main.camera = cam
    main.videoPlayer = vp

    ac._operator_contrast(ContrastImprovement.CLAHE, "CLAHE")

    assert main._contrast_pipeline_methods == []
    assert cam.contrast_pipeline == []
    assert cam.video_handler.processor.config.contrast_pipeline == []
    assert vp.processor.config.contrast_pipeline == []
    cam.set_method_for_contrast.assert_called_once_with(ContrastImprovement.CLAHE)
    vp.set_method_for_contrast.assert_called_once_with(ContrastImprovement.CLAHE)


def test_on_contrast_radio_toggled_ignored_when_unchecked(patched_deps):
    main = make_main_mock()
    ac = AppController(main)
    ac._operator_contrast = MagicMock()

    ac._on_contrast_radio_toggled(False, ContrastImprovement.gamma, "гамма")

    ac._operator_contrast.assert_not_called()


def test_on_contrast_radio_toggled_calls_operator_when_checked(patched_deps):
    main = make_main_mock()
    ac = AppController(main)
    ac._operator_contrast = MagicMock()

    ac._on_contrast_radio_toggled(True, ContrastImprovement.gamma, "гамма")

    ac._operator_contrast.assert_called_once_with(ContrastImprovement.gamma, "гамма")
