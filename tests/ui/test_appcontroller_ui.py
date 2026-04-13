"""UI tests: клики по виджетам при подключённом AppController (зависимости замоканы)."""

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt

from app.core.AppController import AppController
from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.MainWindow import MainWindow


@pytest.fixture
def main_window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def app_controller(main_window):
    with patch("app.core.AppController.CameraManager") as CM, patch(
        "app.core.AppController.FileBrowser"
    ), patch("app.core.AppController.VideoPlayer") as VP:
        cam = MagicMock()
        cam.flag_capture = True
        cam.width = 640
        cam.height = 480
        cam.set_method_for_contrast = MagicMock()
        cam.set_method_for_noise = MagicMock()
        cam.set_monochrome = MagicMock()
        cam.set_record_format = MagicMock()
        CM.return_value.find_cameras = MagicMock()
        CM.return_value.current_camera.return_value = cam
        VP.return_value.is_loaded.return_value = False
        ac = AppController(main_window)
        # Радиокнопки и чекбоксы в отключённых group box не шлют сигналы при клике.
        main_window._set_processing_blocks_enabled(True)
        yield ac, cam


class TestContrastRadiosUi:
    def test_clahe_radio_calls_camera(self, app_controller, qtbot):
        ac, cam = app_controller
        qtbot.mouseClick(ac.main.ui.radio_contrast_clahe, Qt.MouseButton.LeftButton)
        cam.set_method_for_contrast.assert_called_with(ContrastImprovement.CLAHE)

    def test_not_improve_radio(self, app_controller):
        ac, cam = app_controller
        # «Нет» включён по умолчанию — сначала другой режим, затем возврат.
        ac.main.ui.radio_contrast_clahe.click()
        cam.set_method_for_contrast.reset_mock()
        ac.main.ui.radio_contrast_none.click()
        cam.set_method_for_contrast.assert_called_once_with(ContrastImprovement.NotImprove)

    def test_he_radio(self, app_controller, qtbot):
        ac, cam = app_controller
        qtbot.mouseClick(ac.main.ui.radio_contrast_he, Qt.MouseButton.LeftButton)
        cam.set_method_for_contrast.assert_called_with(ContrastImprovement.HE)


class TestNoiseRadiosUi:
    def test_median_radio(self, app_controller, qtbot):
        ac, cam = app_controller
        qtbot.mouseClick(ac.main.ui.radio_noise_median, Qt.MouseButton.LeftButton)
        cam.set_method_for_noise.assert_called_with(NoiseReduction.MedianBlur)


class TestMonochromeCheckboxUi:
    def test_set_monochrome_checked_notifies_camera(self, app_controller):
        ac, cam = app_controller
        ac.main.ui.check_monochrome.setChecked(False)
        cam.set_monochrome.reset_mock()
        ac.main.ui.check_monochrome.setChecked(True)
        cam.set_monochrome.assert_called_once_with(True)
