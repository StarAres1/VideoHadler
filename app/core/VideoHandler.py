from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time

class VideoHandler(QObject):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    paint_frame = pyqtSignal(QPixmap)

    @pyqtSlot()
    def run(self):
        try:
            while self.camera.flag_capture:
                ret, rgb_frame = self.camera.cap.read()

                if not ret:
                    time.sleep(0.01)
                    continue

                rgb_frame = cv2.flip(rgb_frame, 1)

                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = self.camera.channel * self.camera.width

                q_image = QImage(frame.data, self.camera.width, self.camera.height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)

                self.paint_frame.emit(pixmap)

                #TODO: Заменить на сигнал и запись в другом потоке
                """
                if self.flag_record:
                    self.write_video(rgb_frame)
                """
        except Exception as e:
            print(e)
        finally:
            self.camera.disconnect()
