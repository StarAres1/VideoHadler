import cv2
from app.core.VideoHandler import VideoHandler
from app.core.Recorder import Recorder
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSlot, QObject
from PyQt5.QtGui import QPixmap
from app.core.Enums import ContrastImprovement

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
        self.worker_show = None

        self.thread_write = QThread()
        self.worker_write = None

        self.flag_capture = None
        self.flag_record = None

        self.method_for_contrast = ContrastImprovement.NotImprove

    def set_method_for_contrast(self, method):
        self.method_for_contrast = method

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

        self.worker_show = VideoHandler(self)
        self.worker_show.moveToThread(self.thread_show)

        self.worker_show.paint_frame.connect(self.display_frame)
        self.thread_show.started.connect(self.worker_show.run)
        self.worker_show.show_fps.connect(self.show_fps)

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
        self.worker_show.record_frame.connect(self.worker_write.record)
        self.worker_show.close_file.connect(self.worker_write.close_file)

        self.thread_write.start()



    def stop_record(self):
        self.flag_record = False

        #FIXME: должно вызываться сингалом, задержка - костыль, чтобы слот
        # close в Recorder успел выполниться
        self.thread_write.wait(1000)
        self.thread_write.quit()
