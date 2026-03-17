import cv2
from app.core.VideoHandler import VideoHandler
from app.core.Recorder import Recorder
from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSlot, QObject
from PyQt6.QtGui import QPixmap
from app.core.Enums import ContrastImprovement, NoiseReduction

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

        self.method_for_contrast = ContrastImprovement.NotImprove
        self.method_for_noise = NoiseReduction.NotReduction

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
        self.flag_capture = False

        if self.thread_show.isRunning():
            self.thread_show.quit()

    @pyqtSlot(QPixmap)
    def display_frame(self, pixmap):
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
        return True

    def start_record(self):
        self.flag_record = True

        self.worker_write = Recorder(self.width, self.height, self.fps, self.name)
        self.worker_write.moveToThread(self.thread_write)

        self.thread_write.started.connect(self.worker_write.open_file)
        self.video_handler.record_frame.connect(self.worker_write.record)
        self.video_handler.close_file.connect(self.worker_write.close_file)

        self.thread_write.start()


    def stop_record(self):
        self.flag_record = False

        #FIXME: должно вызываться сингалом, задержка - костыль, чтобы слот
        # close в Recorder успел выполниться
        self.thread_write.wait(1000)
        self.thread_write.quit()

    @pyqtSlot(float)
    def set_clipLimit_CLAHE(self, clipLimit):
        self.video_handler.clipLimit = clipLimit

    @pyqtSlot(int)
    def set_titleGridSize_CLAHE(self, titleGridSize):
        self.video_handler.titleGridSize = titleGridSize

    @pyqtSlot(float)
    def set_alpha_adjust(self, alpha):
        self.video_handler.alpha = alpha

    @pyqtSlot(int)
    def set_beta_adjust(self, beta):
        self.video_handler.beta = beta
