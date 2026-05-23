"""Unit tests: Camera (mocked OpenCV / threads)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.Camera import Camera
from app.core.Enums import ContrastImprovement


@pytest.fixture
def camera():
    vf = MagicMock()
    ui = MagicMock()
    return Camera(0, "TestCam", vf, ui)


class TestCameraRoi:
    def test_normalize_roi_positive(self, camera):
        camera.width = 100
        camera.height = 80
        camera.roi_x = 10
        camera.roi_y = 20
        camera.roi_width = 50
        camera.roi_height = 40
        camera._normalize_roi()
        assert camera.roi_x == 10
        assert camera.roi_y == 20

    def test_normalize_roi_clamps_negative(self, camera):
        camera.width = 20
        camera.height = 20
        camera.roi_x = 0
        camera.roi_y = 100
        camera.roi_width = 500
        camera.roi_height = 500
        camera._normalize_roi()
        assert camera.roi_x == 0
        assert camera.roi_y == 0
        assert camera.roi_width == 20
        assert camera.roi_height == 20

    def test_normalize_roi_no_size_early_exit(self, camera):
        camera.width = None
        camera.height = None
        camera.roi_x = 5
        camera._normalize_roi()
        assert camera.roi_x == 5


class TestCameraSetters:
    def test_set_method_for_contrast(self, camera):
        camera.set_method_for_contrast(ContrastImprovement.HE)
        assert camera.method_for_contrast == ContrastImprovement.HE

    def test_set_record_format_lowercase(self, camera):
        camera.set_record_format("MP4")
        assert camera.record_format == "mp4"

    def test_set_record_format_none_defaults(self, camera):
        camera.set_record_format(None)
        assert camera.record_format == "avi"


class TestGetProperty:
    def test_positive_reads_frame(self, camera):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap.get.return_value = 30.0
        camera.cap = mock_cap
        assert camera.get_property() is True
        assert camera.width == 80
        assert camera.height == 60

    def test_negative_not_opened(self, camera):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        camera.cap = mock_cap
        assert camera.get_property() is False

    def test_negative_read_fails(self, camera):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        camera.cap = mock_cap
        assert camera.get_property() is False


class TestStartCapture:
    def test_negative_get_property_fails_releases(self, camera):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("app.core.Camera.cv2.VideoCapture", return_value=mock_cap):
            camera.thread_show = MagicMock()
            camera.thread_show.isRunning.return_value = False
            result = camera.start_capture()
        assert result is False
        mock_cap.release.assert_called()


"""Extended unit tests for Camera.py targeting ~100% coverage."""
import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np
import cv2
from app.core.Camera import Camera
from app.core.Enums import ContrastImprovement, NoiseReduction


# ==================== FIXTURES ====================
@pytest.fixture
def camera():
    """Base camera fixture with mocked UI and video frame."""
    vf = MagicMock()
    ui = MagicMock()
    ui.status_bar = MagicMock()
    return Camera(0, "TestCam", vf, ui)


@pytest.fixture
def camera_with_handler(camera):
    """Camera with mocked video_handler and processor config."""
    camera.video_handler = MagicMock()
    camera.video_handler.processor.config = MagicMock()
    return camera


# ==================== LIFECYCLE & CAPTURE ====================
class TestCaptureLifecycle:
    @patch('app.core.Camera.VideoHandler')
    @patch('app.core.Camera.cv2.VideoCapture')
    def test_start_capture_success(self, mock_cap_cls, mock_vh_cls, camera):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.return_value = 30.0
        mock_cap_cls.return_value = mock_cap

        camera.thread_show = MagicMock()
        camera.thread_show.isRunning.return_value = False

        result = camera.start_capture()
        assert camera.flag_capture is True
        assert camera.cap is mock_cap
        camera.thread_show.start.assert_called_once()

    def test_stop_capture(self, camera):
        camera.flag_capture = True
        camera.flag_record = True
        camera.thread_show = MagicMock()
        camera.thread_show.isRunning.return_value = True
        camera.stop_record = MagicMock()

        camera.stop_capture()
        assert camera.flag_capture is False
        camera.stop_record.assert_called_once()
        camera.thread_show.quit.assert_called_once()
        camera.thread_show.wait.assert_called_once_with(500)

    def test_disconnect(self, camera):
        mock_cap = MagicMock()
        camera.cap = mock_cap
        camera.disconnect()
        mock_cap.release.assert_called_once()


# ==================== RESOLUTION & PROBING ====================
class TestResolutionAndProbing:
    def test_set_selected_resolution(self, camera):
        camera.set_selected_resolution(1280, 720)
        assert camera.selected_resolution == (1280, 720)

    @patch.object(Camera, '_apply_resolution', return_value=True)
    def test_apply_selected_resolution_runtime(self, mock_apply, camera):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda p: {cv2.CAP_PROP_FRAME_WIDTH: 800, cv2.CAP_PROP_FRAME_HEIGHT: 600}.get(p, 0)
        camera.cap = mock_cap
        camera.set_selected_resolution(800, 600)

        assert camera.apply_selected_resolution_runtime() is True
        assert camera.width == 800
        assert camera.height == 600
        assert camera.prop == 800 / 600

    def test_apply_selected_resolution_runtime_no_cap_or_res(self, camera):
        camera.cap = None
        assert camera.apply_selected_resolution_runtime() is False

        camera.cap = MagicMock()
        camera.selected_resolution = None
        assert camera.apply_selected_resolution_runtime() is False

    def test_set_max_supported_resolution(self, camera):
        camera.cap = MagicMock()
        camera.supported_resolutions = [(1280, 720), (640, 480)]

        camera._set_max_supported_resolution()
        assert camera.selected_resolution == (1280, 720)
        camera.cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def test_set_max_supported_resolution_probes_if_empty(self, camera):
        camera.cap = MagicMock()
        camera.supported_resolutions = []
        camera.probe_supported_resolutions = MagicMock(return_value=[(1920, 1080)])

        camera._set_max_supported_resolution()
        assert camera.selected_resolution == (1920, 1080)

    def test_apply_resolution_static_success(self):
        cap = MagicMock()
        assert Camera._apply_resolution(cap, 640, 480) is True
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def test_apply_resolution_static_exception(self):
        cap = MagicMock()
        cap.set.side_effect = RuntimeError("Hardware error")
        assert Camera._apply_resolution(cap, 640, 480) is False

    @patch('app.core.Camera.cv2.VideoCapture')
    def test_probe_supported_resolutions_for_index(self, mock_cap_cls):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.set.return_value = True
        cap.get.side_effect = lambda p: {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}.get(p, 0)
        mock_cap_cls.return_value = cap

        res = Camera.probe_supported_resolutions_for_index(0)
        assert (1280, 720) in res
        cap.release.assert_called_once()

    @patch('app.core.Camera.cv2.VideoCapture')
    def test_probe_supported_resolutions_for_index_fails(self, mock_cap_cls):
        mock_cap_cls.return_value = MagicMock(isOpened=MagicMock(return_value=False))
        assert Camera.probe_supported_resolutions_for_index(0) == []

    @patch.object(Camera, 'probe_supported_resolutions_for_index', return_value=[(1920, 1080)])
    def test_probe_supported_resolutions(self, mock_probe, camera):
        res = camera.probe_supported_resolutions()
        assert res == [(1920, 1080)]
        assert camera.supported_resolutions == [(1920, 1080)]


# ==================== RECORDING ====================
class TestRecording:
    @patch('app.core.Camera.Recorder')
    def test_start_record(self, mock_rec_cls, camera):
        camera.video_handler = MagicMock()
        camera.flag_capture = True
        camera.thread_write = MagicMock()

        camera.start_record("mp4")
        assert camera.flag_record is True
        assert camera.record_format == "mp4"
        mock_rec_cls.return_value.moveToThread.assert_called_once_with(camera.thread_write)
        camera.thread_write.start.assert_called_once()

    def test_start_record_early_exit(self, camera):
        camera.video_handler = None
        camera.flag_capture = True
        camera.thread_write = MagicMock()
        camera.start_record()
        camera.thread_write.start.assert_not_called()

    def test_stop_record(self, camera):
        camera.video_handler = MagicMock()
        camera.worker_write = MagicMock()
        camera.thread_write = MagicMock()
        camera.flag_record = True

        camera.stop_record()
        assert camera.flag_record is False
        camera.thread_write.quit.assert_called_once()
        assert camera.worker_write is None


# ==================== UI SLOTS & PROCESSING ====================
class TestUISlotsAndProcessing:

    def test_set_contrast_pipeline_without_handler(self, camera):
        camera.video_handler = None
        pipeline = [ContrastImprovement.HE]
        camera.set_contrast_pipeline(pipeline)
        assert camera.contrast_pipeline == pipeline

    def test_display_frame(self, camera):
        pixmap = MagicMock()
        camera.display_frame(pixmap)
        pixmap.scaled.assert_called_once()
        camera.video_frame.setPixmap.assert_called_once()

    def test_display_frame_paused(self, camera):
        camera.preview_paused = True
        pixmap = MagicMock()
        camera.display_frame(pixmap)
        pixmap.scaled.assert_not_called()

    def test_display_frame_none_pixmap(self, camera):
        camera.display_frame(None)
        camera.video_frame.setPixmap.assert_not_called()

    def test_show_fps(self, camera):
        camera.show_fps(25.5)
        camera.ui.status_bar.showMessage.assert_called_once_with("FPS: 25.50")


# ==================== CONFIGURATION SETTERS ====================
class TestConfigSetters:
    @pytest.mark.parametrize("method_name, attr_name, value", [
        ("set_alpha_adjust", "alpha", 1.5),
        ("set_beta_adjust", "beta", 10),
        ("set_gamma_value", "gamma", 2.2),
        ("set_monochrome", "monochrome", True),
        ("set_he_color", "he_color", False),
        ("set_clipLimit_CLAHE", "clip_limit", 3.0),
        ("set_titleGridSize_CLAHE", "tile_grid_size", 8),
        ("set_sigmoid_cutoff", "sigmoid_cutoff", 0.5),
        ("set_sigmoid_gain", "sigmoid_gain", 10.0),
        ("set_auto_gamma_target_brightness", "auto_gamma_target_brightness", 128),
        ("set_auto_gamma_color", "auto_gamma_color", True),
        ("set_nn_skip_frames", "nn_skip_frames", 5),
        ("set_zero_dce_strength", "zero_dce_strength", 0.8),
        ("set_median_ksize", "median_ksize", 3),
        ("set_fast_gaussian_ksize", "fast_gaussian_ksize", 5),
        ("set_fast_gaussian_sigma", "fast_gaussian_sigma", 1.2),
    ])
    def test_all_config_setters(self, camera_with_handler, method_name, attr_name, value):
        getattr(camera_with_handler, method_name)(value)
        assert getattr(camera_with_handler.video_handler.processor.config, attr_name) == value

    def test_config_setters_ignores_if_no_handler(self, camera):
        camera.video_handler = None
        # Should not raise
        camera.set_alpha_adjust(1.5)
        camera.set_gamma_value(2.0)


# ==================== ROI SETTERS & NORMALIZATION ====================
class TestRoiSetters:
    @pytest.mark.parametrize("method_name, attr_name, value", [
        ("set_roi_x", "roi_x", 50),
        ("set_roi_y", "roi_y", 30),
        ("set_roi_width", "roi_width", 200),
        ("set_roi_height", "roi_height", 150),
    ])
    def test_roi_setters_apply_and_normalize(self, camera, method_name, attr_name, value):
        camera.width = 640
        camera.height = 480
        getattr(camera, method_name)(value)
        assert getattr(camera, attr_name) == value
        # _normalize_roi is called internally, verify bounds
        assert 0 <= camera.roi_x <= camera.width
        assert 0 <= camera.roi_y <= camera.height

    def test_normalize_roi_edge_cases(self, camera):
        camera.width = 100
        camera.height = 100
        camera.roi_width = 200  # > width
        camera.roi_height = 50  # < height
        camera.roi_x = 80  # near edge
        camera._normalize_roi()
        assert camera.roi_width == 100
        assert camera.roi_x == 0  # clamped because roi_x + width > camera width