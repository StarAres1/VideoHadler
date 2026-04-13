"""Integration tests: MainWindow constructed with AppController (external deps mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.AppController import AppController
from app.core.MainWindow import MainWindow


@pytest.fixture
def main_window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@patch("app.core.AppController.VideoPlayer")
@patch("app.core.AppController.FileBrowser")
@patch("app.core.AppController.CameraManager")
def test_app_controller_wires_main_window(mock_cm, mock_fb, mock_vp, main_window):
    mgr = mock_cm.return_value
    mgr.find_cameras = MagicMock()
    mgr.current_camera.return_value = None
    vp = mock_vp.return_value
    vp.is_loaded.return_value = False

    ac = AppController(main_window)
    mgr.find_cameras.assert_called()
    assert ac.main is main_window
    mock_fb.assert_called_once_with(main_window.ui.file_tree_view)
