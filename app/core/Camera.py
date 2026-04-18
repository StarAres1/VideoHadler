import logging

import cv2
from app.core.VideoHandler import VideoHandler
from app.core.Recorder import Recorder
from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSlot, QObject
from PyQt6.QtGui import QPixmap
from app.core.Enums import ContrastImprovement, NoiseReduction
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class Camera(QObject):
    RESOLUTION_CANDIDATES = [
        (1920, 1080),
        (1600, 1200),
        (1280, 1024),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]

    def __init__(self, index, name, video_frame, ui):
        super().__init__()
        self.cap = None
        self.index = index
        self.name = name
        self.width = None
        self.height = None
        self.channel = None
        self.fps = None
        self.video_frame = video_frame
        self.ui = ui

        self.thread_show = QThread()
        self.video_handler = None

        self.thread_write = QThread()
        self.worker_write = None

        self.flag_capture = None
        self.flag_record = None
        self.record_format = "avi"

        self.method_for_contrast = ContrastImprovement.NotImprove
        self.method_for_noise = NoiseReduction.NotReduction

        self.prop = None
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = 1
        self.roi_height = 1
        self.show_roi_content = False
        self.supported_resolutions = []
        self.selected_resolution = None

    def set_method_for_contrast(self, method):
        self.method_for_contrast = method

    def set_method_for_noise(self, method):
        self.method_for_noise = method

    def disconnect(self):
        self.cap.release()

    def start_capture(self):

        if self.thread_show.isRunning():
            self.stop_capture()

        logger.info("Камера %s (index=%s): подготовка захвата", self.name, self.index)
        # On Windows, DirectShow is usually more stable than MSMF for webcams.
        # Fallback to default backend if DSHOW is unavailable.
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)

        if self.selected_resolution is not None:
            self._apply_resolution(self.cap, self.selected_resolution[0], self.selected_resolution[1])
        else:
            self._set_max_supported_resolution()

        if not self.get_property():
            if self.cap:
                self.cap.release()
            logger.error("Камера %s: не удалось прочитать первый кадр", self.name)
            return False

        self.flag_capture = True

        self.video_handler = VideoHandler(self)
        self.video_handler.moveToThread(self.thread_show)

        self.video_handler.paint_frame.connect(self.display_frame)
        self.thread_show.started.connect(self.video_handler.run)
        self.video_handler.show_fps.connect(self.show_fps)

        logger.info(
            "Камера %s: запуск потока отображения thread_show=%s",
            self.name,
            self.thread_show,
        )
        self.thread_show.start()

    def set_selected_resolution(self, width: int, height: int):
        self.selected_resolution = (int(width), int(height))

    def apply_selected_resolution_runtime(self):
        """Apply selected resolution while capture is active."""
        if not self.cap or not self.selected_resolution:
            return False
        width, height = self.selected_resolution
        if not self._apply_resolution(self.cap, width, height):
            return False
        got_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        got_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if got_w > 0 and got_h > 0:
            self.width = got_w
            self.height = got_h
            self.prop = self.width / self.height if self.height else None
            self._normalize_roi()
            return True
        return False

    def probe_supported_resolutions(self):
        """Probe camera modes and return unique supported resolutions sorted by area desc."""
        resolutions = self.probe_supported_resolutions_for_index(self.index)
        self.supported_resolutions = resolutions
        return resolutions

    @classmethod
    def probe_supported_resolutions_for_index(cls, camera_index: int):
        # Prefer DirectShow for probing on Windows to match runtime capture behavior.
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap or not cap.isOpened():
            return []

        found = {}
        try:
            for cand_w, cand_h in cls.RESOLUTION_CANDIDATES:
                if not cls._apply_resolution(cap, cand_w, cand_h):
                    continue
                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if got_w > 0 and got_h > 0:
                    found[(got_w, got_h)] = got_w * got_h
        finally:
            cap.release()

        resolutions = sorted(found.keys(), key=lambda wh: (wh[0] * wh[1], wh[0], wh[1]), reverse=True)
        return resolutions

    @staticmethod
    def _apply_resolution(cap, width: int, height: int):
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            return True
        except Exception:
            return False

    def _set_max_supported_resolution(self):
        if not self.cap:
            return
        resolutions = self.supported_resolutions or self.probe_supported_resolutions()
        if resolutions:
            best_w, best_h = resolutions[0]
            self._apply_resolution(self.cap, best_w, best_h)
            self.selected_resolution = (best_w, best_h)

    def stop_capture(self):
        logger.info("Камера %s: остановка захвата", self.name)
        if self.flag_record:
            self.stop_record()
        self.flag_capture = False

        if self.thread_show.isRunning():
            self.thread_show.quit()
            self.thread_show.wait(500)
        logger.debug("Камера %s: поток отображения остановлен", self.name)

    @pyqtSlot(QPixmap)
    def display_frame(self, pixmap):
        if pixmap is not None:
            pixmap = pixmap.scaled(self.video_frame.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.video_frame.setPixmap(pixmap)


    @pyqtSlot(float)
    def show_fps(self, fps):
        self.ui.status_bar.showMessage(f"FPS: {fps:.2f}")

    def get_property(self):
        if not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()

        if not ret:
            return False

        self.height, self.width, self.channel = frame.shape

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.prop = self.width / self.height
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = self.width
        self.roi_height = self.height
        self.show_roi_content = False
        logger.debug("Свойства кадра камеры %s: %sx%s", self.name, self.width, self.height)
        return True

    def _normalize_roi(self):
        if not self.width or not self.height:
            return
        self.roi_width = max(1, min(int(self.roi_width), self.width))
        self.roi_height = max(1, min(int(self.roi_height), self.height))
        max_x = max(0, self.width - self.roi_width)
        max_y = max(0, self.height - self.roi_height)
        self.roi_x = max(0, min(int(self.roi_x), max_x))
        self.roi_y = max(0, min(int(self.roi_y), max_y))

    def start_record(self, video_format="avi"):
        if not self.video_handler or not self.flag_capture:
            return

        logger.info("Камера %s: запись в файл, поток записи thread_write=%s", self.name, self.thread_write)
        self.flag_record = True
        self.record_format = (video_format or "avi").lower()

        self.worker_write = Recorder()
        self.worker_write.moveToThread(self.thread_write)

        self.video_handler.open_file.connect(self.worker_write.open_file)
        self.video_handler.record_frame.connect(self.worker_write.record)
        self.video_handler.close_file.connect(self.worker_write.close_file)

        self.thread_write.start()


    def stop_record(self):
        self.flag_record = False

        self.thread_write.wait(1000)
        self.thread_write.quit()
        self.thread_write.wait(500)

        if self.video_handler and self.worker_write:
            try:
                self.video_handler.open_file.disconnect(self.worker_write.open_file)
                self.video_handler.record_frame.disconnect(self.worker_write.record)
                self.video_handler.close_file.disconnect(self.worker_write.close_file)
            except Exception:
                pass
        self.worker_write = None

    @pyqtSlot(float)
    def set_clipLimit_CLAHE(self, clipLimit):
        if self.video_handler:
            self.video_handler.processor.config.clip_limit = float(clipLimit)

    @pyqtSlot(int)
    def set_titleGridSize_CLAHE(self, titleGridSize):
        if self.video_handler:
            self.video_handler.processor.config.tile_grid_size = int(titleGridSize)

    @pyqtSlot(float)
    def set_alpha_adjust(self, alpha):
        if self.video_handler:
            self.video_handler.processor.config.alpha = float(alpha)

    @pyqtSlot(int)
    def set_beta_adjust(self, beta):
        if self.video_handler:
            self.video_handler.processor.config.beta = int(beta)

    @pyqtSlot(str)
    def set_record_format(self, video_format):
        self.record_format = (video_format or "avi").lower()

    @pyqtSlot(bool)
    def set_he_color(self, value):
        if self.video_handler:
            self.video_handler.processor.config.he_color = bool(value)

    @pyqtSlot(float)
    def set_gamma_value(self, value):
        if self.video_handler:
            self.video_handler.processor.config.gamma = float(value)

    @pyqtSlot(float)
    def set_sigmoid_cutoff(self, value):
        if self.video_handler:
            self.video_handler.processor.config.sigmoid_cutoff = float(value)

    @pyqtSlot(float)
    def set_sigmoid_gain(self, value):
        if self.video_handler:
            self.video_handler.processor.config.sigmoid_gain = float(value)

    @pyqtSlot(int)
    def set_auto_gamma_target_brightness(self, value):
        if self.video_handler:
            self.video_handler.processor.config.auto_gamma_target_brightness = int(value)

    @pyqtSlot(bool)
    def set_auto_gamma_color(self, value):
        if self.video_handler:
            self.video_handler.processor.config.auto_gamma_color = bool(value)

    @pyqtSlot(int)
    def set_nn_skip_frames(self, value):
        if self.video_handler:
            self.video_handler.processor.config.nn_skip_frames = int(value)

    @pyqtSlot(int)
    def set_median_ksize(self, value):
        if self.video_handler:
            self.video_handler.processor.config.median_ksize = int(value)

    @pyqtSlot(int)
    def set_fast_gaussian_ksize(self, value):
        if self.video_handler:
            self.video_handler.processor.config.fast_gaussian_ksize = int(value)

    @pyqtSlot(float)
    def set_fast_gaussian_sigma(self, value):
        if self.video_handler:
            self.video_handler.processor.config.fast_gaussian_sigma = float(value)

    @pyqtSlot(bool)
    def set_monochrome(self, value):
        if self.video_handler:
            self.video_handler.processor.config.monochrome = bool(value)

    @pyqtSlot(int)
    def set_roi_x(self, value):
        self.roi_x = int(value)
        self._normalize_roi()

    @pyqtSlot(int)
    def set_roi_y(self, value):
        self.roi_y = int(value)
        self._normalize_roi()

    @pyqtSlot(int)
    def set_roi_width(self, value):
        self.roi_width = int(value)
        self._normalize_roi()

    @pyqtSlot(int)
    def set_roi_height(self, value):
        self.roi_height = int(value)
        self._normalize_roi()
