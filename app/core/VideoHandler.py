from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
from PyQt5.QtCore import QThread
from app.core.Enums import ContrastImprovement
from app.core.ContrastImprover import ContrastImprover

class VideoHandler(QObject):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.flag_open_file = False


    paint_frame = pyqtSignal(QPixmap)
    open_file = pyqtSignal(int, int, int, str)
    record_frame = pyqtSignal(QImage)
    close_file = pyqtSignal()

    @pyqtSlot()
    def run(self):
        flag_try_close = True
        try:
            while self.camera.flag_capture:
                ret, rgb_frame = self.camera.cap.read()

                if self.camera.flag_record and not self.flag_open_file:
                    self.open_file.emit(self.camera.width, self.camera.height, self.camera.fps, self.camera.name)

                    #FIXME: в будущем класс Recorder должен с помощью сигнала сообщать об успехе открытия файла
                    self.flag_open_file = True

                if not self.camera.flag_record and self.flag_open_file and flag_try_close:
                    flag_try_close = False
                    self.close_file.emit()

                if not ret:
                    time.sleep(0.01)
                    continue

                rgb_frame = cv2.flip(rgb_frame, 1)

                match self.camera.method_for_contrast:
                    case ContrastImprovement.NotImprove:
                        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                    case ContrastImprovement.CLAHE:
                        frame = ContrastImprover.CLAHE(rgb_frame)
                    case _:
                        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

                bytes_per_line = self.camera.channel * self.camera.width

                q_image = QImage(frame.data, self.camera.width, self.camera.height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)

                self.paint_frame.emit(pixmap)

                if self.camera.flag_record and self.flag_open_file:
                    self.record_frame.emit(q_image)

        except Exception as e:
            print(e)
        finally:
            self.camera.disconnect()
