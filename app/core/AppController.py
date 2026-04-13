from app.core.CameraManager import CameraManager
from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.custom_widgets.FileBrowser import FileBrowser
from app.core.MainWindow import MainWindow
from app.core.custom_widgets.SpinBox_Slider import SpinBox_Slider
from app.core.VideoPlayer import VideoPlayer


class AppController:
    def __init__(self, main_window: MainWindow):
        self.main = main_window
        self.camera_manager = CameraManager(self.main.ui.video_frame_label, self.main.ui)
        self.current_camera_index = [self.main.ui.combo_cameras.currentIndex()]

        self._init_sources()
        self._init_roi_controls()
        self._connect_ui()

    def _init_sources(self):
        self.camera_manager.find_cameras(self.main.ui.combo_cameras)
        self.main.camera = self.camera_manager.current_camera(0)
        if self.main.camera is None:
            self.main.statusBar().showMessage("Камеры не найдены. Доступен только режим воспроизведения файла.", 5000)

        self.main.tree = FileBrowser(self.main.ui.file_tree_view)
        self.main.videoPlayer = VideoPlayer(self.main.ui.video_frame_label)
        self.main.bind_video_player(self.main.videoPlayer)

    def _set_sources_roi_display(self, active: bool):
        if self.main.camera:
            self.main.camera.show_roi_content = bool(active)
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            self.main.videoPlayer.show_roi_content = bool(active)
            self.main.videoPlayer.refresh_current_frame()

    def _on_toggle_roi_display(self, checked: bool):
        self._set_sources_roi_display(checked)
        self.main.set_roi_content_display_active(checked)

    def _apply_to_active_sources(self, method_name, value):
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

    def _stop_camera_pipeline(self):
        if self.main.camera and self.main.camera.flag_record:
            self.main.stop_camera_recording()
        if self.main.camera and self.main.camera.flag_capture:
            self.main.camera.stop_capture()
        self.main.set_camera_stream_active(False)

    def _stop_file_playback(self):
        if self.main.videoPlayer and self.main.videoPlayer.is_loaded():
            self.main.videoPlayer.stop()
        self.main.ui.playback_group.setVisible(False)
        self.main.ui.label_playback_time.setText("0:00:00")
        if not (self.main.camera and self.main.camera.flag_capture):
            self.main.set_camera_stream_active(False)

    def _on_camera_changed(self, index):
        if index < 0:
            return
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
            self._init_roi_controls_values()
            self.current_camera_index[0] = index

    def _toggle_capture(self):
        if not self.main.camera:
            return
        if self.main.camera.flag_capture:
            self._stop_camera_pipeline()
            return

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

    def _update_roi_limits(self):
        cam = self.main.camera
        if not cam and not (self.main.videoPlayer and self.main.videoPlayer.is_loaded()):
            return
        frame_w = cam.width if cam and cam.width else (self.main.videoPlayer.width if self.main.videoPlayer else 640)
        frame_h = cam.height if cam and cam.height else (self.main.videoPlayer.height if self.main.videoPlayer else 480)
        x = self.main.ui.spin_roi_x.value()
        y = self.main.ui.spin_roi_y.value()
        max_x = max(0, frame_w - 1)
        max_y = max(0, frame_h - 1)
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
        max_w = max(1, frame_w - x)
        max_h = max(1, frame_h - y)
        self.main.ui.slider_roi_w.setMinimum(1)
        self.main.ui.spin_roi_w.setMinimum(1)
        self.main.ui.slider_roi_h.setMinimum(1)
        self.main.ui.spin_roi_h.setMinimum(1)
        self.main.ui.slider_roi_w.setMaximum(max_w)
        self.main.ui.spin_roi_w.setMaximum(max_w)
        self.main.ui.slider_roi_h.setMaximum(max_h)
        self.main.ui.spin_roi_h.setMaximum(max_h)
        if self.main.ui.spin_roi_w.value() > max_w:
            self.main.ui.spin_roi_w.setValue(max_w)
        if self.main.ui.spin_roi_h.value() > max_h:
            self.main.ui.spin_roi_h.setValue(max_h)

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
        self.main.ui.button_refresh_cameras.clicked.connect(lambda: self.camera_manager.find_cameras(self.main.ui.combo_cameras))
        self.main.ui.combo_cameras.currentIndexChanged.connect(self._on_camera_changed)
        self.main.ui.button_toggle_capture.clicked.connect(self._toggle_capture)
        self.main.ui.button_toggle_recording.clicked.connect(self.main.toggle_camera_recording)
        self.main.ui.combo_record_format.currentTextChanged.connect(
            lambda fmt: self.main.camera.set_record_format(fmt) if self.main.camera else None
        )
        self.main.ui.radio_contrast_clahe.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.CLAHE))
        self.main.ui.radio_contrast_none.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.NotImprove))
        self.main.ui.radio_contrast_adjust.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.adjust_contrast))
        self.main.ui.radio_contrast_he.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.HE))
        self.main.ui.radio_contrast_gamma.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.gamma))
        self.main.ui.radio_contrast_sigmoid.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.sigmoid))
        self.main.ui.radio_contrast_nn.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.nn))
        self.main.ui.radio_contrast_nn.clicked.connect(self.main.ensure_nn_model_loaded_async)
        self.main.ui.radio_contrast_auto_gamma.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.autoGamma))
        self.main.ui.radio_noise_none.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_noise", NoiseReduction.NotReduction))
        self.main.ui.radio_noise_median.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_noise", NoiseReduction.MedianBlur))
        self.main.ui.radio_noise_fast_gaussian.clicked.connect(lambda: self._apply_to_active_sources("set_method_for_noise", NoiseReduction.FastGaussian))
        self.main.ui.check_monochrome.toggled.connect(lambda v: self._apply_to_active_sources("set_monochrome", v))
        self.main.ui.button_clahe_info.clicked.connect(self.main.show_dialog_CLAHE)
        self.main.ui.button_adjust_info.clicked.connect(self.main.show_dialog_adjustContrast)
        self.main.ui.button_gamma_info.clicked.connect(self.main.show_gamma_info)
        self.main.ui.button_sigmoid_info.clicked.connect(self.main.show_sigmoid_info)
        self.main.ui.button_nn_auto_info.clicked.connect(self.main.show_nn_auto_info)
        self.main.ui.button_auto_gamma_info.clicked.connect(self.main.show_auto_gamma_info)
        self.main.ui.button_noise_median_info.clicked.connect(self.main.show_noise_median_info)
        self.main.ui.button_noise_fast_gaussian_info.clicked.connect(self.main.show_noise_nlm_info)
        self.main.tree.video_selected.connect(self._on_video_selected)
        self.main.ui.button_toggle_playback.clicked.connect(self.main.toggle_video_playback)
        self.main.ui.button_seek_backward.clicked.connect(self.main.video_seek_backward)
        self.main.ui.button_seek_forward.clicked.connect(self.main.video_seek_forward)
        self.main.ui.button_take_screenshot.clicked.connect(self.main.make_video_screenshot)
        self.main.ui.slider_playback_position.valueChanged.connect(self.main.set_video_position)
        self.main.ui.button_toggle_roi_display.toggled.connect(self._on_toggle_roi_display)
