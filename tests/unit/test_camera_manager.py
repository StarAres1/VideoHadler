"""Unit tests: CameraManager with mocked DirectShow graph."""

from unittest.mock import MagicMock, patch

from app.core.CameraManager import CameraManager


@patch("app.core.CameraManager.FilterGraph")
def test_find_cameras_populates_list(mock_fg):
    devices = ["Cam A", "Cam B"]
    mock_fg.return_value.get_input_devices.return_value = devices
    vf = MagicMock()
    ui = MagicMock()
    mgr = CameraManager(vf, ui)
    gui_list = MagicMock()
    mgr.find_cameras(gui_list)
    assert len(mgr.cameras) == 2
    assert gui_list.addItem.call_count == 2


@patch("app.core.CameraManager.FilterGraph")
def test_current_camera_negative_empty(mock_fg):
    mock_fg.return_value.get_input_devices.return_value = []
    mgr = CameraManager(MagicMock(), MagicMock())
    mgr.cameras.clear()
    assert mgr.current_camera(0) is None


@patch("app.core.CameraManager.FilterGraph")
def test_current_camera_index_out_of_range_uses_first(mock_fg):
    mock_fg.return_value.get_input_devices.return_value = ["Only"]
    mgr = CameraManager(MagicMock(), MagicMock())
    gui = MagicMock()
    mgr.find_cameras(gui)
    cam = mgr.current_camera(99)
    assert cam is mgr.cameras[0]
