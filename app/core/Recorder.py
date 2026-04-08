import os
import time

from PyQt6.QtCore import QObject, pyqtSlot
from datetime import datetime
from PyQt6.QtGui import QImage
import cv2
import numpy as np

class Recorder(QObject):
    def __init__(self):
        super().__init__()
        self.output = None
        self.fourcc = None
        self.filename = None

        self.name_camera = ""
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.video_format = "avi"


    @pyqtSlot(int, int, float, str, str)
    def open_file(self, width, height, fps, name, video_format):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.name_camera = name
        self.video_format = (video_format or "avi").lower()

        os.makedirs("video_output", exist_ok=True)

        if self.video_format == "mp4":
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            extension = "mp4"
        else:
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            extension = "avi"

        self.filename = f'video_output/{self.name_camera.replace(" ", "_")}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.{extension}'
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
            ptr.setsize(frame.sizeInBytes())
            frame = np.array(ptr).reshape(frame.height(), frame.width(), 3)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.output.write(frame_bgr)

    @pyqtSlot()
    def close_file(self):
        if self.output:
            self.output.release()
            self.output = None
            print(f"Запись завершена в {time.time()}")
