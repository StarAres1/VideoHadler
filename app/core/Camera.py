import cv2
from datetime import datetime
from app.core.VideoHandler import VideoHandler
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QPixmap

class Camera(QObject):
    def __init__(self, index, name, video_frame):
        super().__init__()
        self.cap = None
        self.index = index
        self.name = name
        self.flag_record = False
        self.fourcc = None
        self.width = None
        self.height = None
        self.channel = None
        self.output = None
        self.fps = None
        self.video_frame = video_frame
        self.thread_show = QThread()
        self.worker_show = None
        self.flag_capture = None

        self.thread_write = QThread()


    def disconnect(self):
        if self.flag_record:
            self.stop_record()

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

        self.thread_show.start()

    def stop_capture(self):
        self.flag_capture = False

        if self.thread_show.isRunning():
            self.thread_show.quit()

    @pyqtSlot(QPixmap)
    def display_frame(self, pixmap):
        self.video_frame.setPixmap(pixmap)


    def start_record(self):
        self.flag_record = True

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # для формата avi
        filename = f'video_output/{self.name}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.avi'
        self.output = cv2.VideoWriter(
            filename.replace(" ", "_"),
            self.fourcc,
            self.fps,
            (self.width, self.height))

    def stop_record(self):
        self.flag_record = False
        self.fourcc = None

        self.output.release()
        self.output = None

    def write_video(self, frame):
        if self.output is None:
            return

        self.output.write(frame)

    def get_property(self):
        if not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()

        if not ret:
            return False

        self.height, self.width, self.channel = frame.shape

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return True