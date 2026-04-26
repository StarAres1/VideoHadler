import logging
import os
from datetime import datetime

import numpy as np
from PyQt6.QtCore import pyqtSlot, QTimer, QEvent, QRect, QSize
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QMainWindow
from forms.main_window_ui import Ui_MainWindow
from app.neural_network.nn_loader_ui import ensure_nn_model_loaded_async
from app.core.Enums import ContrastImprovement
from app.core.ui_panels import processing_dialogs
from app.core.ui_panels.frame_statistics_metrics import IqaMetricsCalculator, show_frame_statistics_dialog

logger = logging.getLogger(__name__)

_ROI_METHOD_NAMES = frozenset({"set_roi_x", "set_roi_y", "set_roi_width", "set_roi_height"})


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
        self.roi_display_applied_callback = None
        self._roi_content_display_active = False
        self._rb_origin = None
        self._init_roi_overlay_frames()
        self.nn_loader_thread = None
        self.nn_loader_worker = None
        self.nn_progress_dialog = None
        self.zero_dce_loader_thread = None
        self.zero_dce_loader_worker = None
        self.zero_dce_progress_dialog = None
        self.zero_dce_progress_label = None
        self.zero_dce_progress_bar = None
        self._iqa_calc = IqaMetricsCalculator()
        self._frame_stats_dialog = None
        self._frame_stats_resume_file = False
        self._frame_stats_camera_was_paused = False
        self.resolution_radio_buttons = []
        self.resolution_selected_callback = None
        self._contrast_pipeline_dialog = None
        self._contrast_pipeline_methods = []
        self._setup_static_ui()

    def _setup_static_ui(self):
        self.ui.slider_playback_position.setMinimum(0)
        self.ui.slider_playback_position.setMaximum(100)
        self.ui.slider_playback_position.setValue(0)
        self.ui.combo_record_format.clear()
        self.ui.combo_record_format.addItems(["avi", "mp4"])
        self.ui.combo_record_format.setCurrentText("avi")
        self.ui.label_record_time_value.setText("0:00:00")
        self.ui.button_toggle_recording.setText("Запись в файл")
        self.ui.button_seek_backward.setText("-10 сек")
        self.ui.button_seek_forward.setText("+10 сек")
        self.bt_disable_roi = QtWidgets.QPushButton("Отключить ROI", self.ui.roi_group)
        self.ui.gridLayout_6.addWidget(self.bt_disable_roi, 5, 0, 1, 4)
        self.bt_disable_roi.clicked.connect(self.disable_roi)
        self.button_frame_stats = QtWidgets.QPushButton("Статистика по кадру", self.ui.view_mode_group)
        self.ui.view_mode_layout.addWidget(self.button_frame_stats)
        self.button_frame_stats.clicked.connect(lambda: show_frame_statistics_dialog(self))
        self.radio_contrast_pipeline = QtWidgets.QRadioButton("Цепочка методов", self.ui.contrast_group)
        self.ui.main_layout.addWidget(self.radio_contrast_pipeline, 10, 0, 1, 1)
        self.button_contrast_pipeline_info = QtWidgets.QToolButton(self.ui.contrast_group)
        self.button_contrast_pipeline_info.setText("...")
        self.ui.main_layout.addWidget(self.button_contrast_pipeline_info, 10, 1, 1, 1)
        self.radio_contrast_zero_dce = QtWidgets.QRadioButton("Zero-DCE", self.ui.contrast_group)
        self.ui.main_layout.addWidget(self.radio_contrast_zero_dce, 11, 0, 1, 1)
        self.button_zero_dce_info = QtWidgets.QToolButton(self.ui.contrast_group)
        self.button_zero_dce_info.setText("...")
        self.ui.main_layout.addWidget(self.button_zero_dce_info, 11, 1, 1, 1)
        self.ui.video_frame_label.installEventFilter(self)
        self.ui.video_frame_label.setMouseTracking(True)
        self._set_processing_blocks_enabled(False)
        self.ui.playback_group.setVisible(False)
        self.ui.playback_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.ui.playback_group.setFixedHeight(self.ui.playback_group.sizeHint().height())
        self._init_resolution_controls()

    @staticmethod
    def _method_param_title(method: ContrastImprovement) -> str:
        titles = {
            ContrastImprovement.CLAHE: "Параметры CLAHE",
            ContrastImprovement.adjust_contrast: "Параметры линейного преобразования",
            ContrastImprovement.gamma: "Параметры гамма-коррекции",
            ContrastImprovement.sigmoid: "Параметры сигмоидной коррекции",
            ContrastImprovement.autoGamma: "Параметры автогаммы",
            ContrastImprovement.nn: "Параметры нейросетевого метода",
            ContrastImprovement.zero_dce: "Параметры Zero-DCE",
        }
        return titles.get(method, "Параметры метода")

    def _configure_pipeline_method(self, method: ContrastImprovement):
        if method == ContrastImprovement.CLAHE:
            processing_dialogs.show_dialog_clahe(self, self._apply_to_active_sources)
        elif method == ContrastImprovement.adjust_contrast:
            processing_dialogs.show_dialog_adjust_contrast(self, self._apply_to_active_sources)
        elif method == ContrastImprovement.gamma:
            processing_dialogs.show_gamma_info(self, self._apply_to_active_sources)
        elif method == ContrastImprovement.sigmoid:
            processing_dialogs.show_sigmoid_info(self, self._apply_to_active_sources)
        elif method == ContrastImprovement.autoGamma:
            processing_dialogs.show_auto_gamma_info(self, self._apply_to_active_sources)
        elif method == ContrastImprovement.nn:
            processing_dialogs.show_nn_auto_info(self, self._apply_to_active_sources)
        elif method == ContrastImprovement.zero_dce:
            processing_dialogs.show_zero_dce_info(self, self._apply_to_active_sources)

    def _init_resolution_controls(self):
        self.resolution_group = QtWidgets.QGroupBox("Разрешение камеры", self.ui.group_camera_capture)
        self.resolution_layout = QtWidgets.QVBoxLayout(self.resolution_group)
        self.resolution_layout.setContentsMargins(8, 6, 8, 6)
        self.resolution_layout.setSpacing(4)
        self.resolution_hint = QtWidgets.QLabel("Выберите разрешение перед запуском захвата")
        self.resolution_layout.addWidget(self.resolution_hint)
        self.ui.gridLayout_2.addWidget(self.resolution_group, 3, 0, 1, 2)

    def set_camera_resolution_options(self, resolutions, selected, on_selected):
        self.resolution_selected_callback = on_selected
        for rb in self.resolution_radio_buttons:
            self.resolution_layout.removeWidget(rb)
            rb.deleteLater()
        self.resolution_radio_buttons = []
        self.ui.button_toggle_capture.setEnabled(True)

        if not resolutions:
            self.resolution_hint.setText("Не удалось определить поддерживаемые разрешения")
            return

        self.resolution_hint.setText("Доступные разрешения:")
        selected_tuple = tuple(selected) if selected else tuple(resolutions[0])
        for width, height in resolutions:
            rb = QtWidgets.QRadioButton(f"{width} x {height}")
            rb.setChecked((width, height) == selected_tuple)
            rb.toggled.connect(
                lambda checked, w=width, h=height: self._on_resolution_radio_toggled(
                    checked, w, h
                )
            )
            self.resolution_layout.addWidget(rb)
            self.resolution_radio_buttons.append(rb)

    def set_camera_resolution_loading(self):
        for rb in self.resolution_radio_buttons:
            self.resolution_layout.removeWidget(rb)
            rb.deleteLater()
        self.resolution_radio_buttons = []
        self.resolution_hint.setText("Поиск поддерживаемых разрешений...")
        self.ui.button_toggle_capture.setEnabled(False)

    def _on_resolution_radio_toggled(self, checked: bool, width: int, height: int):
        if not checked:
            return
        if self.resolution_selected_callback:
            self.resolution_selected_callback(width, height)

    def bind_video_player(self, video_player):
        self.videoPlayer = video_player
        self.videoPlayer.position_changed.connect(self._on_video_position_changed)
        self.videoPlayer.time_changed.connect(self.ui.label_playback_time.setText)
        self.videoPlayer.playback_state_changed.connect(self._on_playback_state_changed)
        self.videoPlayer.file_opened.connect(self._on_file_opened)
        self.videoPlayer.show_fps.connect(self.show_file_fps)

    @pyqtSlot(float)
    def show_file_fps(self, fps):
        self.ui.status_bar.showMessage(f"FPS (файл): {fps:.2f}")

    @pyqtSlot(str)
    def _on_file_opened(self, file_path: str):
        logger.info("Плеер: файл готов к воспроизведению «%s»", file_path)
        self.ui.label_selected_file_name.setText(os.path.basename(file_path))
        self._set_processing_blocks_enabled(True)
        self.ui.playback_group.setVisible(True)
        self._reset_roi_display_mode_ui()

    def set_camera_stream_active(self, active: bool):
        logger.info("Режим потока камеры: активен=%s", active)
        self._set_processing_blocks_enabled(active)
        self.ui.button_toggle_capture.setText("Стоп" if active else "Старт")
        if active:
            self._reset_roi_display_mode_ui()
            self.ui.playback_group.setVisible(False)
            self.ui.button_toggle_playback.setText("Старт")
            self.ui.label_playback_time.setText("0:00:00")
        elif not (self.videoPlayer and self.videoPlayer.is_loaded()):
            self._set_processing_blocks_enabled(False)
        if not active:
            self.stop_record_timer()
            self.ui.button_toggle_recording.setText("Запись в файл")

    def _set_processing_blocks_enabled(self, enabled: bool):
        self.ui.recording_group.setEnabled(enabled)
        self.ui.contrast_group.setEnabled(enabled)
        self.ui.noise_reduction_group.setEnabled(enabled)
        self.ui.roi_group.setEnabled(enabled)

    def set_roi_controls(self, roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin):
        self.roi_controls = (roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin)

    def set_roi_change_callback(self, callback):
        self.roi_change_callback = callback

    def active_native_frame_size(self) -> tuple[int, int]:
        """Размер кадра в пикселях активного источника: только захват с камеры или только файл."""
        if self.camera is not None and bool(getattr(self.camera, "flag_capture", False)):
            w, h = self.camera.width, self.camera.height
            if w and h:
                return int(w), int(h)
        if self.videoPlayer is not None and self.videoPlayer.is_loaded():
            w, h = self.videoPlayer.width, self.videoPlayer.height
            if w and h:
                return int(w), int(h)
        return (0, 0)

    def _roi_overlay_parent_widget(self):
        """Родитель рамок — не сам QLabel: дочерние виджеты в QLabel сдвигают отрисовку pixmap вверх."""
        lbl = self.ui.video_frame_label
        return lbl.parentWidget() if lbl.parentWidget() is not None else lbl

    def _map_roi_rect_label_to_overlay_parent(self, rect: QRect) -> QRect:
        lbl = self.ui.video_frame_label
        parent = self._roi_overlay_parent_widget()
        if parent is lbl:
            return rect
        top_left = lbl.mapToParent(rect.topLeft())
        return QRect(top_left.x(), top_left.y(), rect.width(), rect.height())

    def _init_roi_overlay_frames(self):
        style = "QFrame { background: transparent; border: 2px solid #00AA00; }"
        overlay_parent = self._roi_overlay_parent_widget()
        self._roi_box_frame = QtWidgets.QFrame(overlay_parent)
        self._roi_box_frame.setStyleSheet(style)
        self._roi_box_frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._roi_box_frame.hide()
        self._roi_drag_frame = QtWidgets.QFrame(overlay_parent)
        self._roi_drag_frame.setStyleSheet(style)
        self._roi_drag_frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._roi_drag_frame.hide()

    def _frame_pixel_size(self):
        return self.active_native_frame_size()

    def _video_pixmap_rect_in_label(self):
        pix = self.ui.video_frame_label.pixmap()
        if pix is None or pix.isNull():
            return None
        label_w = max(1, self.ui.video_frame_label.width())
        label_h = max(1, self.ui.video_frame_label.height())
        pix_w = pix.width()
        pix_h = pix.height()
        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2
        return QRect(offset_x, offset_y, pix_w, pix_h)

    def _clamp_rect_to_video_area(self, rect: QRect) -> QRect:
        area = self._video_pixmap_rect_in_label()
        if area is None:
            return rect.normalized()
        return rect.intersected(area).normalized()

    def _update_roi_toggle_button_text(self):
        if not hasattr(self.ui, "button_toggle_roi_display"):
            return
        if self.ui.button_toggle_roi_display.isChecked():
            self.ui.button_toggle_roi_display.setText("Показать весь кадр")
        else:
            self.ui.button_toggle_roi_display.setText("Показать область ROI на видео")

    def set_roi_content_display_active(self, active: bool):
        self._roi_content_display_active = bool(active)
        self._roi_drag_frame.hide()
        self._rb_origin = None
        self._update_roi_toggle_button_text()
        self.refresh_roi_overlay()

    def refresh_roi_overlay(self):
        if not getattr(self, "_roi_box_frame", None):
            return
        if self._roi_content_display_active:
            self._roi_box_frame.hide()
            return
        fw, fh = self._frame_pixel_size()
        if fw < 1 or fh < 1 or not self.roi_controls:
            self._roi_box_frame.hide()
            return
        rx, ry, rw, rh = (
            self.roi_controls[0].value(),
            self.roi_controls[1].value(),
            self.roi_controls[2].value(),
            self.roi_controls[3].value(),
        )
        if rx == 0 and ry == 0 and rw == fw and rh == fh:
            self._roi_box_frame.hide()
            return
        area = self._video_pixmap_rect_in_label()
        if area is None or area.width() < 1 or area.height() < 1:
            self._roi_box_frame.hide()
            return
        left = area.x() + int((rx / fw) * area.width())
        top = area.y() + int((ry / fh) * area.height())
        w_pix = max(1, int((rw / fw) * area.width()))
        h_pix = max(1, int((rh / fh) * area.height()))
        geo = QRect(left, top, w_pix, h_pix).intersected(area)
        if geo.width() < 2 or geo.height() < 2:
            self._roi_box_frame.hide()
            return
        self._roi_box_frame.setGeometry(self._map_roi_rect_label_to_overlay_parent(geo))
        self._roi_box_frame.show()
        self._roi_box_frame.raise_()

    def _reset_roi_display_mode_ui(self):
        if hasattr(self.ui, "button_toggle_roi_display"):
            self.ui.button_toggle_roi_display.blockSignals(True)
            self.ui.button_toggle_roi_display.setChecked(False)
            self.ui.button_toggle_roi_display.blockSignals(False)
        if self.roi_display_applied_callback:
            self.roi_display_applied_callback(False)
        self.set_roi_content_display_active(False)

    def disable_roi(self):
        logger.info("Оператор: сброс ROI на полный кадр (кнопка)")
        width, height = self.active_native_frame_size()
        if not width or not height or not self.roi_controls:
            return
        if hasattr(self.ui, "button_toggle_roi_display"):
            self.ui.button_toggle_roi_display.blockSignals(True)
            self.ui.button_toggle_roi_display.setChecked(False)
            self.ui.button_toggle_roi_display.blockSignals(False)
        if self.roi_display_applied_callback:
            self.roi_display_applied_callback(False)
        self.set_roi_content_display_active(False)
        roi_x_spin, roi_y_spin, roi_w_spin, roi_h_spin = self.roi_controls
        roi_x_spin.setValue(0)
        roi_y_spin.setValue(0)
        roi_w_spin.setValue(width)
        roi_h_spin.setValue(height)
        if self.roi_change_callback:
            self.roi_change_callback()
        self.refresh_roi_overlay()

    def _apply_roi_from_mouse_rect(self, rect: QRect):
        width, height = self.active_native_frame_size()
        if not width or not height or not self.roi_controls:
            return
        if rect.width() < 4 or rect.height() < 4:
            return
        pix = self.ui.video_frame_label.pixmap()
        if pix is None or pix.isNull():
            return
        label_w = max(1, self.ui.video_frame_label.width())
        label_h = max(1, self.ui.video_frame_label.height())
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
        roi_w_spin.setValue(w)
        roi_h_spin.setValue(h)
        roi_x_spin.setValue(x)
        roi_y_spin.setValue(y)
        if self.roi_change_callback:
            self.roi_change_callback()
        self.refresh_roi_overlay()
        logger.info("Оператор: ROI с мыши x=%s y=%s w=%s h=%s (кадр %sx%s)", x, y, w, h, width, height)

    def eventFilter(self, obj, event):
        if obj is self.ui.video_frame_label:
            if event.type() == QEvent.Type.Resize:
                self.refresh_roi_overlay()
        active_camera = self.camera and self.camera.flag_capture
        active_file = self.videoPlayer and self.videoPlayer.is_loaded()
        if obj is self.ui.video_frame_label and (active_camera or active_file):
            if self._roi_content_display_active:
                return super().eventFilter(obj, event)
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._rb_origin = event.pos()
                self._roi_drag_frame.setGeometry(self._map_roi_rect_label_to_overlay_parent(QRect(self._rb_origin, QSize())))
                self._roi_drag_frame.show()
                self._roi_drag_frame.raise_()
                return True
            if event.type() == QEvent.Type.MouseMove and self._rb_origin is not None:
                r = QRect(self._rb_origin, event.pos()).normalized()
                r = self._clamp_rect_to_video_area(r)
                self._roi_drag_frame.setGeometry(self._map_roi_rect_label_to_overlay_parent(r))
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self._rb_origin is not None:
                    rect = QRect(self._rb_origin, event.pos()).normalized()
                    rect = self._clamp_rect_to_video_area(rect)
                    self._roi_drag_frame.hide()
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
        self.ui.label_record_time_value.setText(self._format_hms(self.record_elapsed_sec))

    def start_record_timer(self):
        self.record_elapsed_sec = 0
        self.ui.label_record_time_value.setText("0:00:00")
        self.record_timer.start(1000)

    def stop_record_timer(self):
        self.record_timer.stop()
        self.ui.label_record_time_value.setText("0:00:00")

    @pyqtSlot()
    def start_camera_recording(self):
        if not self.camera:
            self.statusBar().showMessage("Сначала выберите камеру", 2500)
            logger.info("Запись: камера не выбрана")
            return
        if not self.camera.flag_capture:
            self.statusBar().showMessage("Сначала запустите захват кадров", 2500)
            logger.info("Запись: захват не запущен")
            return
        selected_format = self.ui.combo_record_format.currentText().strip().lower() or "avi"
        logger.info("Оператор: начало записи с камеры, формат=%s", selected_format)
        self.camera.set_record_format(selected_format)
        self.camera.start_record(selected_format)
        self.start_record_timer()
        self.ui.button_toggle_recording.setText("Завершить запись")

    @pyqtSlot()
    def stop_camera_recording(self):
        logger.info("Оператор: остановка записи с камеры")
        if self.camera:
            self.camera.stop_record()
        self.stop_record_timer()
        self.ui.button_toggle_recording.setText("Запись в файл")

    @pyqtSlot()
    def toggle_camera_recording(self):
        logger.info("Оператор: кнопка записи с камеры")
        if not self.camera or not self.camera.flag_capture:
            self.start_camera_recording()
            return
        if self.camera.flag_record:
            self.stop_camera_recording()
        else:
            self.start_camera_recording()

    @pyqtSlot(int)
    def _on_video_position_changed(self, value: int):
        self.ui.slider_playback_position.blockSignals(True)
        self.ui.slider_playback_position.setValue(value)
        self.ui.slider_playback_position.blockSignals(False)

    @pyqtSlot(bool)
    def _on_playback_state_changed(self, is_playing: bool):
        self.ui.button_toggle_playback.setText("Пауза" if is_playing else "Старт")
        if self.videoPlayer and self.videoPlayer.is_loaded():
            self.ui.playback_group.setVisible(True)
            self.ui.playback_group.setEnabled(True)
            self._set_processing_blocks_enabled(True)

    @pyqtSlot()
    def toggle_video_playback(self):
        if not self.videoPlayer or not self.videoPlayer.is_loaded():
            self.statusBar().showMessage("Сначала выберите видеофайл", 2500)
            logger.info("Оператор: пауза/воспроизведение — нет загруженного файла")
            return
        logger.info("Оператор: переключение паузы/воспроизведения файла")
        self.videoPlayer.toggle_play_pause()

    @pyqtSlot()
    def video_seek_backward(self):
        logger.info("Оператор: перемотка файла на -10 с")
        if self.videoPlayer:
            self.videoPlayer.seek_seconds(-10)

    @pyqtSlot()
    def video_seek_forward(self):
        logger.info("Оператор: перемотка файла на +10 с")
        if self.videoPlayer:
            self.videoPlayer.seek_seconds(10)

    @pyqtSlot(int)
    def set_video_position(self, value: int):
        logger.debug("Оператор: позиция воспроизведения %s%%", value)
        if self.videoPlayer:
            self.videoPlayer.set_position_percent(value)

    @pyqtSlot()
    def make_video_screenshot(self):
        logger.info("Оператор: сохранение скриншота области предпросмотра")
        pixmap = self.ui.video_frame_label.pixmap()
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

    def _get_current_frame_pair_for_statistics(self):
        if self.camera and getattr(self.camera, "flag_capture", False):
            before = getattr(self.camera, "last_preview_before_rgb", None)
            after = getattr(self.camera, "last_preview_after_rgb", None)
            if before is not None and after is not None:
                return before.copy(), after.copy(), "camera"
        if self.videoPlayer and self.videoPlayer.is_loaded():
            before, after = self.videoPlayer.get_frame_pair_for_statistics()
            if before is not None and after is not None:
                return before, after, "file"
        return None, None, None

    def _pause_preview_for_stats(self, source: str):
        self._frame_stats_resume_file = False
        self._frame_stats_camera_was_paused = False
        if source == "camera" and self.camera:
            self._frame_stats_camera_was_paused = bool(getattr(self.camera, "preview_paused", False))
            self.camera.preview_paused = True
            logger.info("Статистика кадра: предпросмотр камеры приостановлен")
            return
        if source == "file" and self.videoPlayer and self.videoPlayer.is_loaded():
            self._frame_stats_resume_file = self.videoPlayer.is_playing()
            if self._frame_stats_resume_file:
                self.videoPlayer.pause()
            logger.info("Статистика кадра: воспроизведение файла приостановлено")

    @staticmethod
    def _contrast_method_display_name(method: ContrastImprovement) -> str:
        mapping = {
            ContrastImprovement.NotImprove: "Без улучшения",
            ContrastImprovement.CLAHE: "CLAHE",
            ContrastImprovement.adjust_contrast: "Линейное преобразование",
            ContrastImprovement.HE: "Эквализация гистограммы (HE)",
            ContrastImprovement.gamma: "Гамма-коррекция",
            ContrastImprovement.autoGamma: "Автогамма",
            ContrastImprovement.sigmoid: "Сигмоидная коррекция",
            ContrastImprovement.nn: "Автоподбор нейросетью",
            ContrastImprovement.pipeline: "Цепочка методов",
            ContrastImprovement.zero_dce: "Zero-DCE",
        }
        return mapping.get(method, str(method))

    def _contrast_method_with_params(self, method: ContrastImprovement, processor) -> str:
        cfg = processor.config
        name = self._contrast_method_display_name(method)
        if method == ContrastImprovement.CLAHE:
            return f"{name} (clipLimit={cfg.clip_limit:.2f}, tileGrid={cfg.tile_grid_size})"
        if method == ContrastImprovement.adjust_contrast:
            return f"{name} (alpha={cfg.alpha:.2f}, beta={cfg.beta})"
        if method == ContrastImprovement.gamma:
            return f"{name} (gamma={cfg.gamma:.2f})"
        if method == ContrastImprovement.autoGamma:
            return f"{name} (target_brightness={cfg.auto_gamma_target_brightness})"
        if method == ContrastImprovement.sigmoid:
            return f"{name} (cutoff={cfg.sigmoid_cutoff:.2f}, gain={cfg.sigmoid_gain:.2f})"
        if method == ContrastImprovement.nn:
            nn_label = getattr(processor, "_nn_last_label", "") or "не определён"
            return f"{name} (skip_frames={cfg.nn_skip_frames}, выбранный метод={nn_label})"
        if method == ContrastImprovement.zero_dce:
            return f"{name} (strength={cfg.zero_dce_strength:.2f})"
        return name

    def _build_applied_contrast_text(self, source: str) -> str:
        if source == "camera" and self.camera:
            processor = self.camera.video_handler.processor if self.camera.video_handler else None
            method = self.camera.method_for_contrast
            pipeline = list(getattr(self.camera, "contrast_pipeline", []) or [])
        elif source == "file" and self.videoPlayer:
            processor = self.videoPlayer.processor
            method = self.videoPlayer.method_for_contrast
            pipeline = list(getattr(self.videoPlayer.processor.config, "contrast_pipeline", []) or [])
        else:
            return "Источник не определён."

        if processor is None:
            return "Обработчик кадров недоступен."

        if method == ContrastImprovement.pipeline:
            if not pipeline:
                return "Режим: цепочка методов. Цепочка пустая."
            chain = [self._contrast_method_with_params(m, processor) for m in pipeline]
            return "Режим: цепочка методов.\nПоследовательность:\n" + "\n".join(f"{idx+1}. {entry}" for idx, entry in enumerate(chain))

        return "Режим: одиночный метод.\n" + self._contrast_method_with_params(method, processor)

    def _resume_preview_after_stats(self):
        if self.camera and getattr(self.camera, "flag_capture", False):
            self.camera.preview_paused = self._frame_stats_camera_was_paused
        if self._frame_stats_resume_file and self.videoPlayer and self.videoPlayer.is_loaded():
            self.videoPlayer.resume()
        self._frame_stats_resume_file = False
        self._frame_stats_camera_was_paused = False

    def ask_user_confirmation(self, title: str, text: str) -> bool:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        yes_btn = box.addButton("Да", QtWidgets.QMessageBox.ButtonRole.YesRole)
        no_btn = box.addButton("Нет", QtWidgets.QMessageBox.ButtonRole.NoRole)
        box.setDefaultButton(no_btn)
        box.exec()
        return box.clickedButton() is yes_btn

    def _apply_to_active_sources(self, method_name: str, value):
        targets = []
        if self.camera and getattr(self.camera, "flag_capture", False):
            targets.append("camera")
        if self.videoPlayer and self.videoPlayer.is_loaded():
            targets.append("file")
        tgt = "+".join(targets) if targets else "—"
        if method_name in _ROI_METHOD_NAMES:
            logger.debug("Диалог: параметр %s=%r -> [%s]", method_name, value, tgt)
        else:
            logger.info("Диалог: параметр %s=%r -> [%s]", method_name, value, tgt)

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
