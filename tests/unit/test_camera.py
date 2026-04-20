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
