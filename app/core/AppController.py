import logging

from app.core.CameraManager import CameraManager
from app.core.Camera import Camera
from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.custom_widgets.FileBrowser import FileBrowser
from app.core.MainWindow import MainWindow
from app.core.custom_widgets.SpinBox_Slider import SpinBox_Slider
from app.core.VideoPlayer import VideoPlayer
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QTimer

logger = logging.getLogger(__name__)

_ROI_METHOD_NAMES = frozenset({"set_roi_x", "set_roi_y", "set_roi_width", "set_roi_height"})


class ResolutionProbeWorker(QObject):
    finished = pyqtSignal(int, object)

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index

    @pyqtSlot()
    def run(self):
        logger.info(
            "Фоновый опрос разрешений: старт camera_index=%s",
            self.camera_index,
        )
        try:
            resolutions = Camera.probe_supported_resolutions_for_index(self.camera_index)
        except Exception:
            logger.exception("Фоновый опрос разрешений: ошибка camera_index=%s", self.camera_index)
            resolutions = []
        logger.info(
            "Фоновый опрос разрешений: завершён camera_index=%s, найдено режимов=%s",
            self.camera_index,
            len(resolutions),
        )
        self.finished.emit(self.camera_index, resolutions)


class AppController:
    def __init__(self, main_window: MainWindow):
        self.main = main_window
        self.camera_manager = CameraManager(self.main.ui.video_frame_label, self.main.ui)
        self.current_camera_index = [self.main.ui.combo_cameras.currentIndex()]
        self._resolution_probe_thread = None
        self._resolution_probe_worker = None
        self._resolution_probe_request_id = 0
        self._active_probe_request_id = 0
        self._pending_probe_camera_index = None

        self._init_sources()
        self._init_roi_controls()
        self._connect_ui()
        logger.info("AppController: инициализация завершена")

    def _init_sources(self):
        self.camera_manager.find_cameras(self.main.ui.combo_cameras)
        self.main.camera = self.camera_manager.current_camera(0)
        if self.main.camera is None:
            self.main.statusBar().showMessage("Камеры не найдены. Доступен только режим воспроизведения файла.", 5000)
        else:
            self._refresh_resolution_options()

        self.main.tree = FileBrowser(self.main.ui.file_tree_view)
        self.main.videoPlayer = VideoPlayer(self.main.ui.video_frame_label)
        self.main.bind_video_player(self.main.videoPlayer)
        logger.info("Источники: камеры обнаружены в combo, видеоплеер подключён")

    def _set_sources_roi_display(self, active: bool):
        if self.main.camera:
            self.main.camera.show_roi_content = bool(active)
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            self.main.videoPlayer.show_roi_content = bool(active)
            self.main.videoPlayer.refresh_current_frame()

    def _on_toggle_roi_display(self, checked: bool):
        logger.info("Оператор: режим отображения ROI на видео=%s", checked)
        self._set_sources_roi_display(checked)
        self.main.set_roi_content_display_active(checked)

    def _apply_to_active_sources(self, method_name, value):
        targets = []
        if self.main.camera and getattr(self.main.camera, "flag_capture", False):
            targets.append("camera")
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            targets.append("file")
        tgt = "+".join(targets) if targets else "—"
        logger.debug("Параметр обработки %s=%r -> источники [%s]", method_name, value, tgt)

        if self.main.camera and getattr(self.main.camera, "flag_capture", False) and hasattr(self.main.camera, method_name):
            if value is None:
                getattr(self.main.camera, method_name)()
            else:
                getattr(self.main.camera, method_name)(value)
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded() and hasattr(self.main.videoPlayer, method_name):
            if value is None:
                getattr(self.main.videoPlayer, method_name)()
            else:
                getattr(self.main.videoPlayer, method_name)(value)

    def _operator_contrast(self, method: ContrastImprovement, label: str) -> None:
        logger.info("Оператор: метод улучшения контраста «%s»", label)
        self._apply_to_active_sources("set_method_for_contrast", method)

    def _operator_noise(self, method: NoiseReduction, label: str) -> None:
        logger.info("Оператор: метод шумоподавления «%s»", label)
        self._apply_to_active_sources("set_method_for_noise", method)

    def _operator_monochrome(self, checked: bool) -> None:
        logger.info("Оператор: монохромный режим=%s", checked)
        self._apply_to_active_sources("set_monochrome", checked)

    def _on_record_format_changed(self, fmt: str) -> None:
        logger.info("Оператор: формат записи видео «%s»", fmt)
        if self.main.camera:
            self.main.camera.set_record_format(fmt)

    def _open_dialog_clahe_info(self) -> None:
        logger.info("Оператор: диалог настроек CLAHE")
        self.main.show_dialog_CLAHE()

    def _open_dialog_adjust_info(self) -> None:
        logger.info("Оператор: диалог линейной коррекции контраста")
        self.main.show_dialog_adjustContrast()

    def _open_dialog_gamma_info(self) -> None:
        logger.info("Оператор: диалог гамма-коррекции")
        self.main.show_gamma_info()

    def _open_dialog_sigmoid_info(self) -> None:
        logger.info("Оператор: диалог сигмоидной коррекции")
        self.main.show_sigmoid_info()

    def _open_dialog_nn_info(self) -> None:
        logger.info("Оператор: диалог авто-нейроконтраста")
        self.main.show_nn_auto_info()

    def _open_dialog_auto_gamma_info(self) -> None:
        logger.info("Оператор: диалог автогаммы")
        self.main.show_auto_gamma_info()

    def _open_dialog_median_noise_info(self) -> None:
        logger.info("Оператор: диалог медианного фильтра")
        self.main.show_noise_median_info()

    def _open_dialog_fast_gauss_info(self) -> None:
        logger.info("Оператор: диалог быстрого гаусса")
        self.main.show_noise_nlm_info()

    def _stop_camera_pipeline(self):
        logger.info("Остановка контура камеры (запись/захват)")
        if self.main.camera and self.main.camera.flag_record:
            self.main.stop_camera_recording()
        if self.main.camera and self.main.camera.flag_capture:
            self.main.camera.stop_capture()
        self.main.set_camera_stream_active(False)

    def _stop_file_playback(self):
        logger.info("Остановка воспроизведения файла")
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            self.main.videoPlayer.stop()
        self.main.ui.playback_group.setVisible(False)
        self.main.ui.label_playback_time.setText("0:00:00")
        if not (self.main.camera and self.main.camera.flag_capture):
            self.main.set_camera_stream_active(False)

    def _on_camera_changed(self, index):
        if index < 0:
            return
        logger.info("Оператор: смена камеры в списке, индекс=%s", index)
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            approved = self.main.ask_user_confirmation(
                "Переключение на камеру",
                "Сейчас идет воспроизведение видеофайла. Оно будет остановлено. Продолжить?"
            )
            if not approved:
                self.main.ui.combo_cameras.blockSignals(True)
                self.main.ui.combo_cameras.setCurrentIndex(self.current_camera_index[0])
                self.main.ui.combo_cameras.blockSignals(False)
                return
            self._stop_file_playback()

        if self.main.camera and self.main.camera.flag_capture and index != self.current_camera_index[0]:
            approved = self.main.ask_user_confirmation(
                "Смена камеры",
                "Текущий захват с камеры будет остановлен, а запись в файл (если идет) прервана. Продолжить?"
            )
            if not approved:
                self.main.ui.combo_cameras.blockSignals(True)
                self.main.ui.combo_cameras.setCurrentIndex(self.current_camera_index[0])
                self.main.ui.combo_cameras.blockSignals(False)
                return
            self._stop_camera_pipeline()

        selected = self.camera_manager.current_camera(index)
        if selected:
            self.main.camera = selected
            self._refresh_resolution_options()
            self._init_roi_controls_values()
            self.current_camera_index[0] = index

    def _on_resolution_selected(self, width: int, height: int):
        logger.info("Оператор: выбрано разрешение камеры %sx%s", width, height)
        if self.main.camera:
            self.main.camera.set_selected_resolution(width, height)
            if getattr(self.main.camera, "flag_capture", False):
                self.main.camera.apply_selected_resolution_runtime()
                self._init_roi_controls_values()

    def _refresh_resolution_options(self):
        logger.debug("Обновление списка разрешений для текущей камеры")
        if not self.main.camera:
            self.main.set_camera_resolution_options([], None, self._on_resolution_selected)
            self.main.ui.button_toggle_capture.setEnabled(False)
            return
        if not isinstance(self.main.camera, Camera):
            self.main.set_camera_resolution_options([], None, self._on_resolution_selected)
            self.main.ui.button_toggle_capture.setEnabled(False)
            return

        cached = self.camera_manager.get_cached_resolutions(self.main.camera.index)
        if cached:
            logger.info("Разрешения камеры из кэша (%s шт.)", len(cached))
            self.main.camera.supported_resolutions = list(cached)
            selected = self.main.camera.selected_resolution or cached[0]
            self.main.camera.set_selected_resolution(selected[0], selected[1])
            self.main.set_camera_resolution_options(cached, selected, self._on_resolution_selected)
            return

        self.main.set_camera_resolution_loading()
        logger.info("Запуск фонового опроса разрешений камеры index=%s", self.main.camera.index)
        self._start_resolution_probe(self.main.camera.index)

    def _start_resolution_probe(self, camera_index: int):
        if self._resolution_probe_thread is not None:
            try:
                if self._resolution_probe_thread.isRunning():
                    logger.info(
                        "Опрос разрешений занят: запрос камеры index=%s поставлен в очередь",
                        camera_index,
                    )
                    self._pending_probe_camera_index = camera_index
                    return
            except RuntimeError:
                # Underlying C++ QThread can be already deleted.
                self._resolution_probe_thread = None
                self._resolution_probe_worker = None
        self._resolution_probe_request_id += 1
        request_id = self._resolution_probe_request_id
        self._active_probe_request_id = request_id
        self._pending_probe_camera_index = None
        self._resolution_probe_thread = QThread()
        self._resolution_probe_worker = ResolutionProbeWorker(camera_index)
        self._resolution_probe_worker.moveToThread(self._resolution_probe_thread)
        self._resolution_probe_thread.started.connect(self._resolution_probe_worker.run)
        self._resolution_probe_worker.finished.connect(
            lambda camera_idx, resolutions, rid=request_id: self._on_resolution_probe_finished(rid, camera_idx, resolutions)
        )
        self._resolution_probe_worker.finished.connect(self._resolution_probe_thread.quit)
        self._resolution_probe_worker.finished.connect(self._resolution_probe_worker.deleteLater)
        self._resolution_probe_thread.finished.connect(self._on_resolution_probe_thread_finished)
        self._resolution_probe_thread.finished.connect(self._resolution_probe_thread.deleteLater)
        self._resolution_probe_thread.start()
        logger.debug(
            "Поток опроса разрешений запущен request_id=%s camera_index=%s",
            request_id,
            camera_index,
        )

    def _on_resolution_probe_thread_finished(self):
        logger.debug("Поток опроса разрешений уничтожен")
        self._resolution_probe_thread = None
        self._resolution_probe_worker = None

    def _on_resolution_probe_finished(self, request_id: int, camera_index: int, resolutions):
        logger.info(
            "Результат опроса разрешений: request_id=%s camera_index=%s актуальный=%s режимов=%s",
            request_id,
            camera_index,
            request_id == self._active_probe_request_id,
            len(resolutions or []),
        )
        if request_id == self._active_probe_request_id:
            self.camera_manager.set_cached_resolutions(camera_index, resolutions)
            if self.main.camera and self.main.camera.index == camera_index:
                self.main.camera.supported_resolutions = list(resolutions)
                if not resolutions:
                    self.main.set_camera_resolution_options([], None, self._on_resolution_selected)
                else:
                    selected = self.main.camera.selected_resolution or resolutions[0]
                    self.main.camera.set_selected_resolution(selected[0], selected[1])
                    self.main.set_camera_resolution_options(resolutions, selected, self._on_resolution_selected)

        if self._pending_probe_camera_index is not None:
            next_index = self._pending_probe_camera_index
            self._pending_probe_camera_index = None
            logger.info("Планируется опрос разрешений для камеры index=%s", next_index)
            QTimer.singleShot(120, lambda idx=next_index: self._start_resolution_probe(idx))

    def _toggle_capture(self):
        if not self.main.camera:
            return
        if self.main.camera.flag_capture:
            logger.info("Оператор: остановка захвата с камеры")
            self._stop_camera_pipeline()
            return

        logger.info("Оператор: запуск захвата с камеры")
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            approved = self.main.ask_user_confirmation(
                "Переход в режим камеры",
                "Воспроизведение видеофайла будет остановлено. Продолжить?"
            )
            if not approved:
                return
            self._stop_file_playback()

        self.main.camera.start_capture()
        self._init_roi_controls_values()
        self.main.set_camera_stream_active(True)

    def _on_video_selected(self, path):
        logger.info("Оператор: выбран видеофайл «%s»", path)
        if self.main.camera and self.main.camera.flag_capture:
            approved = self.main.ask_user_confirmation(
                "Переход к видеофайлу",
                "Текущий захват с камеры будет остановлен, а запись в файл (если идет) прервана. Продолжить?"
            )
            if not approved:
                return
            self._stop_camera_pipeline()
        self.main.videoPlayer.play(path)

    def _init_roi_controls_values(self):
        cam = self.main.camera
        if not cam:
            return

        frame_w = cam.width if cam.width else 640
        frame_h = cam.height if cam.height else 480
        controls = [
            (self.main.ui.slider_roi_x, self.main.ui.spin_roi_x, frame_w - 1),
            (self.main.ui.slider_roi_y, self.main.ui.spin_roi_y, frame_h - 1),
            (self.main.ui.slider_roi_w, self.main.ui.spin_roi_w, frame_w),
            (self.main.ui.slider_roi_h, self.main.ui.spin_roi_h, frame_h),
        ]
        for slider, spin, max_val in controls:
            slider.setMinimum(0 if max_val > 1 else 1)
            spin.setMinimum(0 if max_val > 1 else 1)
            slider.setMaximum(max_val)
            spin.setMaximum(max_val)

        if cam.width and cam.height:
            self.main.ui.slider_roi_w.setMinimum(1)
            self.main.ui.spin_roi_w.setMinimum(1)
            self.main.ui.slider_roi_h.setMinimum(1)
            self.main.ui.spin_roi_h.setMinimum(1)
            cam.set_roi_x(0)
            cam.set_roi_y(0)
            cam.set_roi_width(cam.width)
            cam.set_roi_height(cam.height)
            self.main.ui.spin_roi_x.setValue(0)
            self.main.ui.spin_roi_y.setValue(0)
            self.main.ui.spin_roi_w.setValue(cam.width)
            self.main.ui.spin_roi_h.setValue(cam.height)
        self._update_roi_limits()

    def _on_video_file_opened(self, _path: str):
        """Новый файл: ROI в плеере уже на весь кадр — синхронизируем спинбоксы (иначе остаются размеры предыдущего ролика)."""
        logger.info("Видеофайл открыт в плеере, синхронизация ROI с размером кадра")
        self._reset_roi_ui_to_full_frame_for_active_source()
        self._update_roi_limits()

    def _reset_roi_ui_to_full_frame_for_active_source(self):
        frame_w, frame_h = self.main.active_native_frame_size()
        if frame_w < 1 or frame_h < 1:
            return
        max_x = max(0, frame_w - 1)
        max_y = max(0, frame_h - 1)
        widgets = (
            self.main.ui.slider_roi_x,
            self.main.ui.spin_roi_x,
            self.main.ui.slider_roi_y,
            self.main.ui.spin_roi_y,
            self.main.ui.slider_roi_w,
            self.main.ui.spin_roi_w,
            self.main.ui.slider_roi_h,
            self.main.ui.spin_roi_h,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self.main.ui.slider_roi_x.setMaximum(max_x)
            self.main.ui.spin_roi_x.setMaximum(max_x)
            self.main.ui.slider_roi_y.setMaximum(max_y)
            self.main.ui.spin_roi_y.setMaximum(max_y)
            self.main.ui.slider_roi_w.setMinimum(1)
            self.main.ui.spin_roi_w.setMinimum(1)
            self.main.ui.slider_roi_h.setMinimum(1)
            self.main.ui.spin_roi_h.setMinimum(1)
            self.main.ui.slider_roi_w.setMaximum(frame_w)
            self.main.ui.spin_roi_w.setMaximum(frame_w)
            self.main.ui.slider_roi_h.setMaximum(frame_h)
            self.main.ui.spin_roi_h.setMaximum(frame_h)
            self.main.ui.slider_roi_x.setValue(0)
            self.main.ui.spin_roi_x.setValue(0)
            self.main.ui.slider_roi_y.setValue(0)
            self.main.ui.spin_roi_y.setValue(0)
            self.main.ui.slider_roi_w.setValue(frame_w)
            self.main.ui.spin_roi_w.setValue(frame_w)
            self.main.ui.slider_roi_h.setValue(frame_h)
            self.main.ui.spin_roi_h.setValue(frame_h)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._apply_to_active_sources("set_roi_x", 0)
        self._apply_to_active_sources("set_roi_y", 0)
        self._apply_to_active_sources("set_roi_width", frame_w)
        self._apply_to_active_sources("set_roi_height", frame_h)
        self.main.refresh_roi_overlay()

    def _update_roi_limits(self):
        cam = self.main.camera
        if not cam and not (self.main.videoPlayer and self.main.videoPlayer.is_loaded()):
            return
        frame_w, frame_h = self.main.active_native_frame_size()
        if frame_w < 1:
            frame_w = 640
        if frame_h < 1:
            frame_h = 480
        x = self.main.ui.spin_roi_x.value()
        y = self.main.ui.spin_roi_y.value()
        roi_w = self.main.ui.spin_roi_w.value()
        roi_h = self.main.ui.spin_roi_h.value()
        roi_w = max(1, min(roi_w, frame_w))
        roi_h = max(1, min(roi_h, frame_h))
        if self.main.ui.spin_roi_w.value() != roi_w:
            self.main.ui.spin_roi_w.setValue(roi_w)
        if self.main.ui.spin_roi_h.value() != roi_h:
            self.main.ui.spin_roi_h.setValue(roi_h)

        max_x = max(0, frame_w - roi_w)
        max_y = max(0, frame_h - roi_h)
        self.main.ui.slider_roi_x.setMaximum(max_x)
        self.main.ui.spin_roi_x.setMaximum(max_x)
        self.main.ui.slider_roi_y.setMaximum(max_y)
        self.main.ui.spin_roi_y.setMaximum(max_y)
        if x > max_x:
            self.main.ui.spin_roi_x.setValue(max_x)
            x = max_x
        if y > max_y:
            self.main.ui.spin_roi_y.setValue(max_y)
            y = max_y
        self.main.ui.slider_roi_w.setMinimum(1)
        self.main.ui.spin_roi_w.setMinimum(1)
        self.main.ui.slider_roi_h.setMinimum(1)
        self.main.ui.spin_roi_h.setMinimum(1)
        self.main.ui.slider_roi_w.setMaximum(frame_w)
        self.main.ui.spin_roi_w.setMaximum(frame_w)
        self.main.ui.slider_roi_h.setMaximum(frame_h)
        self.main.ui.spin_roi_h.setMaximum(frame_h)

    def _init_roi_controls(self):
        self.main.roi_display_applied_callback = self._set_sources_roi_display
        self.main.sl_sp_roi_x = SpinBox_Slider(
            self.main.ui.slider_roi_x, self.main.ui.spin_roi_x, lambda v: self._apply_to_active_sources("set_roi_x", v), 0, 0
        )
        self.main.sl_sp_roi_y = SpinBox_Slider(
            self.main.ui.slider_roi_y, self.main.ui.spin_roi_y, lambda v: self._apply_to_active_sources("set_roi_y", v), 0, 0
        )
        self.main.sl_sp_roi_w = SpinBox_Slider(
            self.main.ui.slider_roi_w, self.main.ui.spin_roi_w, lambda v: self._apply_to_active_sources("set_roi_width", v), 1, 1
        )
        self.main.sl_sp_roi_h = SpinBox_Slider(
            self.main.ui.slider_roi_h, self.main.ui.spin_roi_h, lambda v: self._apply_to_active_sources("set_roi_height", v), 1, 1
        )
        self.main.set_roi_controls(self.main.ui.spin_roi_x, self.main.ui.spin_roi_y, self.main.ui.spin_roi_w, self.main.ui.spin_roi_h)
        self.main.set_roi_change_callback(self._update_roi_limits)
        self.main.ui.spin_roi_x.valueChanged.connect(lambda _: self._update_roi_limits())
        self.main.ui.spin_roi_y.valueChanged.connect(lambda _: self._update_roi_limits())
        self.main.ui.spin_roi_w.valueChanged.connect(lambda _: self._update_roi_limits())
        self.main.ui.spin_roi_h.valueChanged.connect(lambda _: self._update_roi_limits())
        for w in (
            self.main.ui.spin_roi_x,
            self.main.ui.spin_roi_y,
            self.main.ui.spin_roi_w,
            self.main.ui.spin_roi_h,
        ):
            w.valueChanged.connect(self.main.refresh_roi_overlay)
        self._init_roi_controls_values()

    def _connect_ui(self):
        self.main.ui.button_refresh_cameras.clicked.connect(self._refresh_cameras_and_resolutions)
        self.main.ui.combo_cameras.currentIndexChanged.connect(self._on_camera_changed)
        self.main.ui.button_toggle_capture.clicked.connect(self._toggle_capture)
        self.main.ui.button_toggle_recording.clicked.connect(self.main.toggle_camera_recording)
        self.main.ui.combo_record_format.currentTextChanged.connect(self._on_record_format_changed)
        self.main.ui.radio_contrast_clahe.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.CLAHE, "CLAHE")
        )
        self.main.ui.radio_contrast_none.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.NotImprove, "без улучшения")
        )
        self.main.ui.radio_contrast_adjust.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.adjust_contrast, "линейная коррекция")
        )
        self.main.ui.radio_contrast_he.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.HE, "выравнивание гистограммы")
        )
        self.main.ui.radio_contrast_gamma.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.gamma, "гамма")
        )
        self.main.ui.radio_contrast_sigmoid.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.sigmoid, "сигмоида")
        )
        self.main.ui.radio_contrast_nn.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.nn, "нейросеть (классификация)")
        )
        self.main.ui.radio_contrast_nn.clicked.connect(self.main.ensure_nn_model_loaded_async)
        self.main.ui.radio_contrast_auto_gamma.clicked.connect(
            lambda: self._operator_contrast(ContrastImprovement.autoGamma, "автогамма")
        )
        self.main.ui.radio_noise_none.clicked.connect(
            lambda: self._operator_noise(NoiseReduction.NotReduction, "без шумоподавления")
        )
        self.main.ui.radio_noise_median.clicked.connect(
            lambda: self._operator_noise(NoiseReduction.MedianBlur, "медианный фильтр")
        )
        self.main.ui.radio_noise_fast_gaussian.clicked.connect(
            lambda: self._operator_noise(NoiseReduction.FastGaussian, "быстрый гаусс")
        )
        self.main.ui.check_monochrome.toggled.connect(self._operator_monochrome)
        self.main.ui.button_clahe_info.clicked.connect(self._open_dialog_clahe_info)
        self.main.ui.button_adjust_info.clicked.connect(self._open_dialog_adjust_info)
        self.main.ui.button_gamma_info.clicked.connect(self._open_dialog_gamma_info)
        self.main.ui.button_sigmoid_info.clicked.connect(self._open_dialog_sigmoid_info)
        self.main.ui.button_nn_auto_info.clicked.connect(self._open_dialog_nn_info)
        self.main.ui.button_auto_gamma_info.clicked.connect(self._open_dialog_auto_gamma_info)
        self.main.ui.button_noise_median_info.clicked.connect(self._open_dialog_median_noise_info)
        self.main.ui.button_noise_fast_gaussian_info.clicked.connect(self._open_dialog_fast_gauss_info)
        self.main.tree.video_selected.connect(self._on_video_selected)
        self.main.ui.button_toggle_playback.clicked.connect(self.main.toggle_video_playback)
        self.main.ui.button_seek_backward.clicked.connect(self.main.video_seek_backward)
        self.main.ui.button_seek_forward.clicked.connect(self.main.video_seek_forward)
        self.main.ui.button_take_screenshot.clicked.connect(self.main.make_video_screenshot)
        self.main.ui.slider_playback_position.valueChanged.connect(self.main.set_video_position)
        self.main.ui.button_toggle_roi_display.toggled.connect(self._on_toggle_roi_display)
        self.main.videoPlayer.file_opened.connect(self._on_video_file_opened)

    def _refresh_cameras_and_resolutions(self):
        logger.info("Оператор: обновление списка камер и разрешений")
        self.camera_manager.find_cameras(self.main.ui.combo_cameras)
        index = self.main.ui.combo_cameras.currentIndex()
        if index >= 0:
            selected = self.camera_manager.current_camera(index)
            if selected:
                self.main.camera = selected
                self.current_camera_index[0] = index
                self._refresh_resolution_options()
