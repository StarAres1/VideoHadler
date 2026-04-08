import cv2
from app.core.VideoHandler import VideoHandler
from app.core.Recorder import Recorder
from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSlot, QObject
from PyQt6.QtGui import QPixmap
from app.core.Enums import ContrastImprovement, NoiseReduction
from PyQt6.QtCore import Qt

class Camera(QObject):
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

    def set_method_for_contrast(self, method):
        self.method_for_contrast = method

    def set_method_for_noise(self, method):
        self.method_for_noise = method

    def disconnect(self):
        self.cap.release()

    def start_capture(self):

        if self.thread_show.isRunning():
            self.stop_capture()

        self.cap = cv2.VideoCapture(self.index)

        if not self.get_property():
            if self.cap:
                self.cap.release()
            return False

        self.flag_capture = True

        self.video_handler = VideoHandler(self)
        self.video_handler.moveToThread(self.thread_show)

        self.video_handler.paint_frame.connect(self.display_frame)
        self.thread_show.started.connect(self.video_handler.run)
        self.video_handler.show_fps.connect(self.show_fps)

        self.thread_show.start()

    def stop_capture(self):
        if self.flag_record:
            self.stop_record()
        self.flag_capture = False

        if self.thread_show.isRunning():
            self.thread_show.quit()
            self.thread_show.wait(500)

    @pyqtSlot(QPixmap)
    def display_frame(self, pixmap):
        if pixmap is not None:
            pixmap = pixmap.scaled(self.video_frame.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.video_frame.setPixmap(pixmap)


    @pyqtSlot(float)
    def show_fps(self, fps):
        self.ui.statusbar.showMessage(f"FPS: {fps:.2f}")

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
        return True

    def _normalize_roi(self):
        if not self.width or not self.height:
            return
        self.roi_x = max(0, min(int(self.roi_x), self.width - 1))
        self.roi_y = max(0, min(int(self.roi_y), self.height - 1))
        self.roi_width = max(1, min(int(self.roi_width), self.width - self.roi_x))
        self.roi_height = max(1, min(int(self.roi_height), self.height - self.roi_y))

    def start_record(self, video_format="avi"):
        if not self.video_handler or not self.flag_capture:
            return

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

        #FIXME: должно вызываться сингалом, задержка - костыль, чтобы слот
        # close в Recorder успел выполниться
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
