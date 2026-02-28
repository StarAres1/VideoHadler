import time

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from datetime import datetime
from PyQt5.QtGui import QImage
import cv2
import numpy as np

class Recorder(QObject):
    def __init__(self, width, height, fps, name):
        super().__init__()
        self.output = None
        self.fourcc = None
        self.filename = None

        self.name_camera = name
        self.width = width
        self.height = height
        self.fps = fps


    @pyqtSlot()
    def open_file(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')   # для формата avi
        self.filename = f'video_output/{self.name_camera.replace(" ", "_")}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.avi'
        print(self.filename)
        self.output = cv2.VideoWriter(
            self.filename,
            self.fourcc,
            self.fps,
            (self.width, self.height))
        print(f"Запись начата в {time.time()}")

    @pyqtSlot(QImage)
    def record(self, frame):
        if self.output:
            # Конвертируем QImage в numpy массив
            # QImage Format_RGB888 -> numpy (H, W, 3)
            ptr = frame.bits()
            ptr.setsize(frame.byteCount())
            frame = np.array(ptr).reshape(frame.height(), frame.width(), 3)

            # OpenCV ожидает BGR, а QImage RGB888 это RGB. Нужно конвертировать.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.output.write(frame_bgr)

    @pyqtSlot()
    def close_file(self):
        print(f"АУ")
        self.output.release()
        print(f"Запись завершена в {time.time()}")
