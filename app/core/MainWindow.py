import os
from datetime import datetime

from PyQt6.QtCore import pyqtSlot, pyqtSignal, QObject, QTimer, QEvent, QRect, QThread
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QMainWindow
from forms.main_window_ui import Ui_MainWindow
from app.core.SpinBox_Slider import SpinBox_Slider

from forms.dialog_clahe_ui import Ui_Dialog as ClaheWindow
from forms.dialog_adjust_contrast_ui import Ui_Dialog as AdjustWindow
from app.core.NNContrastSelector import NN_SELECTOR


class NNModelLoadWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    @pyqtSlot()
    def run(self):
        ok = NN_SELECTOR.ensure_loaded_with_progress(lambda v, t: self.progress.emit(v, t))
        self.finished.emit(ok, NN_SELECTOR.last_error)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.camera = None
        self.videoPlayer = None
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self._update_record_timer)
        self.record_elapsed_sec = 0
        self.roi_change_callback = None
        self.roi_controls = None
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.ui.videoFrame)
        self._rb_origin = None
        self.nn_loader_thread = None
        self.nn_loader_worker = None
        self.nn_progress_dialog = None
        self._setup_static_ui()

    def _setup_static_ui(self):
        self.ui.horizontalSlider_5.setMinimum(0)
        self.ui.horizontalSlider_5.setMaximum(100)
        self.ui.horizontalSlider_5.setValue(0)
        self.ui.c_format.clear()
        self.ui.c_format.addItems(["avi", "mp4"])
        self.ui.c_format.setCurrentText("avi")
        self.ui.label_8.setText("0:00:00")
        self.ui.bt_start_record.setText("Запись в файл")
        self.ui.pushButton_3.setText("-10 сек")
        self.ui.pushButton_2.setText("+10 сек")
        self.bt_disable_roi = QtWidgets.QPushButton("Отключить ROI", self.ui.groupBox_ROI)
        self.ui.gridLayout_6.addWidget(self.bt_disable_roi, 4, 0, 1, 4)
        self.bt_disable_roi.clicked.connect(self.disable_roi)
        self.ui.videoFrame.installEventFilter(self)
        self.ui.videoFrame.setMouseTracking(True)
        self._set_processing_blocks_enabled(False)
        self.ui.groupBox_2.setVisible(False)
        self.ui.groupBox_2.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.ui.groupBox_2.setFixedHeight(self.ui.groupBox_2.sizeHint().height())

    def bind_video_player(self, video_player):
        self.videoPlayer = video_player
        self.videoPlayer.position_changed.connect(self._on_video_position_changed)
        self.videoPlayer.time_changed.connect(self.ui.label_11.setText)
        self.videoPlayer.playback_state_changed.connect(self._on_playback_state_changed)
        self.videoPlayer.file_opened.connect(self._on_file_opened)
        self.videoPlayer.show_fps.connect(self.show_file_fps)

    @pyqtSlot(float)
    def show_file_fps(self, fps):
        self.ui.statusbar.showMessage(f"FPS (файл): {fps:.2f}")

    @pyqtSlot(str)
    def _on_file_opened(self, file_path: str):
        self.ui.label_10.setText(os.path.basename(file_path))
        self._set_processing_blocks_enabled(True)
        self.ui.groupBox_2.setVisible(True)

    def set_camera_stream_active(self, active: bool):
        self._set_processing_blocks_enabled(active)
        self.ui.bt_start_capture.setText("Стоп" if active else "Старт")
        if active:
            self.ui.groupBox_2.setVisible(False)
            self.ui.pushButton.setText("Старт")
            self.ui.label_11.setText("0:00:00")
        elif not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self._set_processing_blocks_enabled(False)
        if not active:
            self.stop_record_timer()
            self.ui.bt_start_record.setText("Запись в файл")

    def _set_processing_blocks_enabled(self, enabled: bool):
        self.ui.groupBox.setEnabled(enabled)
        self.ui.group_contrast_enhancement.setEnabled(enabled)
        self.ui.group_enhancement.setEnabled(enabled)
        self.ui.groupBox_ROI.setEnabled(enabled)

    def set_roi_controls(self, roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin):
        self.roi_controls = (roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin)

    def set_roi_change_callback(self, callback):
        self.roi_change_callback = callback

    def disable_roi(self):
        width = self.camera.width if self.camera and self.camera.width else (self.videoPlayer.width if self.videoPlayer else 0)
        height = self.camera.height if self.camera and self.camera.height else (self.videoPlayer.height if self.videoPlayer else 0)
        if not width or not height or not self.roi_controls:
            return
        roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin = self.roi_controls
        roi_x_spin.setValue(0)
        roi_y_spin.setValue(0)
        roi_w_spin.setValue(width)
        roi_h_spin.setValue(height)
        if self.roi_change_callback:
            self.roi_change_callback()

    def _apply_roi_from_mouse_rect(self, rect: QRect):
        width = self.camera.width if self.camera and self.camera.width else (self.videoPlayer.width if self.videoPlayer else 0)
        height = self.camera.height if self.camera and self.camera.height else (self.videoPlayer.height if self.videoPlayer else 0)
        if not width or not height or not self.roi_controls:
            return
        if rect.width() < 4 or rect.height() < 4:
            return
        pix = self.ui.videoFrame.pixmap()
        if pix is None or pix.isNull():
            return
        label_w = max(1, self.ui.videoFrame.width())
        label_h = max(1, self.ui.videoFrame.height())
        pix_w = pix.width()
        pix_h = pix.height()
        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2

        inter = rect.intersected(QRect(offset_x, offset_y, pix_w, pix_h))
        if inter.isEmpty():
            return

        rel_x = inter.left() - offset_x
        rel_y = inter.top() - offset_y
        rel_w = inter.width()
        rel_h = inter.height()

        x = int((rel_x / max(1, pix_w)) * width)
        y = int((rel_y / max(1, pix_h)) * height)
        w = int((rel_w / max(1, pix_w)) * width)
        h = int((rel_h / max(1, pix_h)) * height)

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin = self.roi_controls
        roi_x_spin.setValue(x)
        roi_y_spin.setValue(y)
        roi_w_spin.setValue(w)
        roi_h_spin.setValue(h)
        if self.roi_change_callback:
            self.roi_change_callback()

    def eventFilter(self, obj, event):
        active_camera = self.camera and self.camera.flag_capture
        active_file = self.videoPlayer and self.videoPlayer.is_loaded()
        if obj is self.ui.videoFrame and (active_camera or active_file):
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._rb_origin = event.pos()
                self._rubber_band.setGeometry(QRect(self._rb_origin, QtCore.QSize()))
                self._rubber_band.show()
                return True
            if event.type() == QEvent.Type.MouseMove and self._rb_origin is not None:
                self._rubber_band.setGeometry(QRect(self._rb_origin, event.pos()).normalized())
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self._rb_origin is not None:
                    rect = QRect(self._rb_origin, event.pos()).normalized()
                    self._rubber_band.hide()
                    self._apply_roi_from_mouse_rect(rect)
                    self._rb_origin = None
                    return True
        return super().eventFilter(obj, event)

    def _format_hms(self, seconds: int) -> str:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        sec = seconds % 60
        return f"{hours}:{minutes:02d}:{sec:02d}"

    def _update_record_timer(self):
        self.record_elapsed_sec += 1
        self.ui.label_8.setText(self._format_hms(self.record_elapsed_sec))

    def start_record_timer(self):
        self.record_elapsed_sec = 0
        self.ui.label_8.setText("0:00:00")
        self.record_timer.start(1000)

    def stop_record_timer(self):
        self.record_timer.stop()
        self.ui.label_8.setText("0:00:00")

    @pyqtSlot()
    def start_camera_recording(self):
        if not self.camera:
            self.statusBar().showMessage("Сначала выберите камеру", 2500)
            return
        if not self.camera.flag_capture:
            self.statusBar().showMessage("Сначала запустите захват кадров", 2500)
            return
        selected_format = self.ui.c_format.currentText().strip().lower() or "avi"
        self.camera.set_record_format(selected_format)
        self.camera.start_record(selected_format)
        self.start_record_timer()
        self.ui.bt_start_record.setText("Завершить запись")

    @pyqtSlot()
    def stop_camera_recording(self):
        if self.camera:
            self.camera.stop_record()
        self.stop_record_timer()
        self.ui.bt_start_record.setText("Запись в файл")

    @pyqtSlot()
    def toggle_camera_recording(self):
        if not self.camera or not self.camera.flag_capture:
            self.start_camera_recording()
            return
        if self.camera.flag_record:
            self.stop_camera_recording()
        else:
            self.start_camera_recording()

    @pyqtSlot(int)
    def _on_video_position_changed(self, value: int):
        self.ui.horizontalSlider_5.blockSignals(True)
        self.ui.horizontalSlider_5.setValue(value)
        self.ui.horizontalSlider_5.blockSignals(False)

    @pyqtSlot(bool)
    def _on_playback_state_changed(self, is_playing: bool):
        self.ui.pushButton.setText("Пауза" if is_playing else "Старт")
        if self.videoPlayer and self.videoPlayer.is_loaded():
            self.ui.groupBox_2.setVisible(True)
            self.ui.groupBox_2.setEnabled(True)
            self._set_processing_blocks_enabled(True)

    @pyqtSlot()
    def toggle_video_playback(self):
        if not self.videoPlayer or not self.videoPlayer.is_loaded():
            self.statusBar().showMessage("Сначала выберите видеофайл", 2500)
            return
        self.videoPlayer.toggle_play_pause()

    @pyqtSlot()
    def video_seek_backward(self):
        if self.videoPlayer:
            self.videoPlayer.seek_seconds(-10)

    @pyqtSlot()
    def video_seek_forward(self):
        if self.videoPlayer:
            self.videoPlayer.seek_seconds(10)

    @pyqtSlot(int)
    def set_video_position(self, value: int):
        if self.videoPlayer:
            self.videoPlayer.set_position_percent(value)

    @pyqtSlot()
    def make_video_screenshot(self):
        pixmap = self.ui.videoFrame.pixmap()
        if pixmap is None or pixmap.isNull():
            self.statusBar().showMessage("Невозможно сделать скриншот: нет активного кадра", 3000)
            return
        output_dir = "screenshots"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(output_dir, filename)
        saved = pixmap.save(path, "PNG")
        if saved:
            self.statusBar().showMessage(f"Скриншот сохранен: {path}", 4000)
        else:
            self.statusBar().showMessage("Не удалось сохранить скриншот", 3000)

    @pyqtSlot()
    def show_noise_median_info(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки медианного фильтра")
        layout = QtWidgets.QGridLayout(dialog)
        label = QtWidgets.QLabel("Размер ядра")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(30)
        spin = QtWidgets.QSpinBox()
        spin.setMinimum(3)
        spin.setMaximum(61)
        layout.addWidget(label, 0, 0)
        layout.addWidget(slider, 0, 1)
        layout.addWidget(spin, 0, 2)
        self.sl_sp_median = SpinBox_Slider(
            slider,
            spin,
            lambda value: self._apply_to_active_sources("set_median_ksize", value),
            1,
            3,
            lambda v: max(3, v * 2 + 1),
            lambda v: max(1, int((v - 1) / 2))
        )
        dialog.show()
        self.dialog_median = dialog

    @pyqtSlot()
    def show_noise_nlm_info(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки быстрого гауссова шумоподавления")
        layout = QtWidgets.QGridLayout(dialog)

        lbl_k = QtWidgets.QLabel("Размер ядра")
        sld_k = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        sld_k.setMinimum(1)
        sld_k.setMaximum(30)
        spn_k = QtWidgets.QSpinBox()
        spn_k.setMinimum(3)
        spn_k.setMaximum(61)

        lbl_sigma = QtWidgets.QLabel("Сигма")
        sld_sigma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        sld_sigma.setMinimum(1)
        sld_sigma.setMaximum(30)
        spn_sigma = QtWidgets.QDoubleSpinBox()
        spn_sigma.setMinimum(0.1)
        spn_sigma.setMaximum(3.0)
        spn_sigma.setSingleStep(0.1)

        layout.addWidget(lbl_k, 0, 0)
        layout.addWidget(sld_k, 0, 1)
        layout.addWidget(spn_k, 0, 2)
        layout.addWidget(lbl_sigma, 1, 0)
        layout.addWidget(sld_sigma, 1, 1)
        layout.addWidget(spn_sigma, 1, 2)

        self.sl_sp_fast_gauss_k = SpinBox_Slider(
            sld_k, spn_k, lambda value: self._apply_to_active_sources("set_fast_gaussian_ksize", value),
            1, 3, lambda v: max(3, v * 2 + 1), lambda v: max(1, int((v - 1) / 2))
        )
        self.sl_sp_fast_gauss_sigma = SpinBox_Slider(
            sld_sigma, spn_sigma, lambda value: self._apply_to_active_sources("set_fast_gaussian_sigma", value),
            10, 1.0, SpinBox_Slider.pow10_int, SpinBox_Slider.dec10_float
        )

        dialog.show()
        self.dialog_fast_gaussian = dialog

    def ask_user_confirmation(self, title: str, text: str) -> bool:
        result = QtWidgets.QMessageBox.question(
            self,
            title,
            text,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        return result == QtWidgets.QMessageBox.StandardButton.Yes

    @pyqtSlot()
    def show_gamma_info(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки гамма-коррекции")
        layout = QtWidgets.QGridLayout(dialog)
        label = QtWidgets.QLabel("Гамма")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(2)
        slider.setMaximum(50)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setMinimum(0.2)
        spin.setMaximum(5.0)
        spin.setSingleStep(0.1)
        layout.addWidget(label, 0, 0)
        layout.addWidget(slider, 0, 1)
        layout.addWidget(spin, 0, 2)
        self.sl_sp_gamma = SpinBox_Slider(
            slider,
            spin,
            lambda value: self._apply_to_active_sources("set_gamma_value", value),
            15,
            1.5,
            SpinBox_Slider.pow10_int,
            SpinBox_Slider.dec10_float
        )
        dialog.show()
        self.dialog_gamma = dialog

    @pyqtSlot()
    def show_sigmoid_info(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки сигмоидной коррекции")
        layout = QtWidgets.QGridLayout(dialog)

        label_cutoff = QtWidgets.QLabel("Отсечка")
        slider_cutoff = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_cutoff.setMinimum(1)
        slider_cutoff.setMaximum(99)
        spin_cutoff = QtWidgets.QDoubleSpinBox()
        spin_cutoff.setMinimum(0.01)
        spin_cutoff.setMaximum(0.99)
        spin_cutoff.setSingleStep(0.01)

        label_gain = QtWidgets.QLabel("Коэффициент")
        slider_gain = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_gain.setMinimum(1)
        slider_gain.setMaximum(30)
        spin_gain = QtWidgets.QSpinBox()
        spin_gain.setMinimum(1)
        spin_gain.setMaximum(30)

        layout.addWidget(label_cutoff, 0, 0)
        layout.addWidget(slider_cutoff, 0, 1)
        layout.addWidget(spin_cutoff, 0, 2)
        layout.addWidget(label_gain, 1, 0)
        layout.addWidget(slider_gain, 1, 1)
        layout.addWidget(spin_gain, 1, 2)

        self.sl_sp_sigmoid_cutoff = SpinBox_Slider(
            slider_cutoff,
            spin_cutoff,
            lambda value: self._apply_to_active_sources("set_sigmoid_cutoff", value),
            50,
            0.5,
            lambda v: int(v * 100),
            lambda v: float(v / 100.0)
        )
        self.sl_sp_sigmoid_gain = SpinBox_Slider(
            slider_gain,
            spin_gain,
            lambda value: self._apply_to_active_sources("set_sigmoid_gain", value),
            12,
            12
        )
        dialog.show()
        self.dialog_sigmoid = dialog

    @pyqtSlot()
    def show_auto_gamma_info(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки авто-гаммы")
        layout = QtWidgets.QGridLayout(dialog)
        label = QtWidgets.QLabel("Целевая яркость")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(254)
        spin = QtWidgets.QSpinBox()
        spin.setMinimum(1)
        spin.setMaximum(254)
        layout.addWidget(label, 0, 0)
        layout.addWidget(slider, 0, 1)
        layout.addWidget(spin, 0, 2)
        self.sl_sp_auto_gamma = SpinBox_Slider(slider, spin, lambda value: self._apply_to_active_sources("set_auto_gamma_target_brightness", value), 128, 128)
        dialog.show()
        self.dialog_auto_gamma = dialog

    @pyqtSlot()
    def show_nn_auto_info(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки автоподбора нейросетью")
        layout = QtWidgets.QGridLayout(dialog)
        label = QtWidgets.QLabel("Пропуск кадров после анализа")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(60)
        spin = QtWidgets.QSpinBox()
        spin.setMinimum(0)
        spin.setMaximum(60)
        layout.addWidget(label, 0, 0)
        layout.addWidget(slider, 0, 1)
        layout.addWidget(spin, 0, 2)
        self.sl_sp_nn_skip = SpinBox_Slider(
            slider,
            spin,
            lambda value: self._apply_to_active_sources("set_nn_skip_frames", value),
            0,
            0
        )
        dialog.show()
        self.dialog_nn_auto = dialog

    def ensure_nn_model_loaded_async(self):
        if NN_SELECTOR.is_loaded() or NN_SELECTOR.is_loading():
            return
        self.nn_progress_dialog = QtWidgets.QDialog(self)
        self.nn_progress_dialog.setWindowTitle("Загрузка нейросети")
        self.nn_progress_dialog.setModal(False)
        layout = QtWidgets.QVBoxLayout(self.nn_progress_dialog)
        self.nn_progress_label = QtWidgets.QLabel("Подготовка...")
        self.nn_progress_bar = QtWidgets.QProgressBar()
        self.nn_progress_bar.setRange(0, 100)
        self.nn_progress_bar.setValue(0)
        layout.addWidget(self.nn_progress_label)
        layout.addWidget(self.nn_progress_bar)
        self.nn_progress_dialog.show()

        self.nn_loader_thread = QThread(self)
        self.nn_loader_worker = NNModelLoadWorker()
        self.nn_loader_worker.moveToThread(self.nn_loader_thread)
        self.nn_loader_thread.started.connect(self.nn_loader_worker.run)
        self.nn_loader_worker.progress.connect(self._on_nn_load_progress)
        self.nn_loader_worker.finished.connect(self._on_nn_load_finished)
        self.nn_loader_worker.finished.connect(self.nn_loader_thread.quit)
        self.nn_loader_worker.finished.connect(self.nn_loader_worker.deleteLater)
        self.nn_loader_thread.finished.connect(self.nn_loader_thread.deleteLater)
        self.nn_loader_thread.start()

    @pyqtSlot(int, str)
    def _on_nn_load_progress(self, value: int, text: str):
        if self.nn_progress_bar:
            self.nn_progress_bar.setValue(value)
        if self.nn_progress_label:
            self.nn_progress_label.setText(text)

    @pyqtSlot(bool, str)
    def _on_nn_load_finished(self, ok: bool, error: str):
        if self.nn_progress_dialog:
            self.nn_progress_dialog.close()
            self.nn_progress_dialog = None
        if not ok:
            self.statusBar().showMessage(f"Ошибка загрузки нейросети: {error}", 5000)

    @pyqtSlot()
    def show_dialog_CLAHE(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        self.dialog_clahe = QtWidgets.QDialog()
        self.ui_dialog_clahe = ClaheWindow()
        self.ui_dialog_clahe.setupUi(self.dialog_clahe)

        self.sl_sp_titleGrid = SpinBox_Slider(self.ui_dialog_clahe.slider_titlleGrid, self.ui_dialog_clahe.spB_titleGrid, lambda value: self._apply_to_active_sources("set_titleGridSize_CLAHE", value),
                                         4, 4, None, None)

        self.sl_sp_clipLimit = SpinBox_Slider(self.ui_dialog_clahe.slider_clipLimit, self.ui_dialog_clahe.spB_clipLimit, lambda value: self._apply_to_active_sources("set_clipLimit_CLAHE", value),
                                         4, 2.0, SpinBox_Slider.pow2_int, SpinBox_Slider.dec2_float)

        self.dialog_clahe.show()

    @pyqtSlot()
    def show_dialog_adjustContrast(self):
        if not self.camera and not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
            return
        self.dialog_adjust = QtWidgets.QDialog()
        self.ui_dialog_adjust = AdjustWindow()
        self.ui_dialog_adjust.setupUi(self.dialog_adjust)

        self.sl_sp_contrast = SpinBox_Slider(self.ui_dialog_adjust.slisder_contrast, self.ui_dialog_adjust.spB_contrast, lambda value: self._apply_to_active_sources("set_alpha_adjust", value),
                                         10, 1.0, SpinBox_Slider.pow10_int, SpinBox_Slider.dec10_float)

        self.sl_sp_brightness = SpinBox_Slider(self.ui_dialog_adjust.slider_brightness, self.ui_dialog_adjust.spB_brightness, lambda value: self._apply_to_active_sources("set_beta_adjust", value),
                                         0, 0, None, None)

        self.dialog_adjust.show()

    def _apply_to_active_sources(self, method_name: str, value):
        if self.camera and getattr(self.camera, "flag_capture", False) and hasattr(self.camera, method_name):
            if value is None:
                getattr(self.camera, method_name)()
            else:
                getattr(self.camera, method_name)(value)
        if self.videoPlayer and self.videoPlayer.is_loaded() and hasattr(self.videoPlayer, method_name):
            if value is None:
                getattr(self.videoPlayer, method_name)()
            else:
                getattr(self.videoPlayer, method_name)(value)
