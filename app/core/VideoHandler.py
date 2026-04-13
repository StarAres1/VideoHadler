from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import cv2
import time
from app.core.FrameProcessor import FrameProcessor

class VideoHandler(QObject):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.flag_open_file = False

        self.current_frame = 0.0
        self.current_time = 0.0

        self.processor = FrameProcessor()


    paint_frame = pyqtSignal(QPixmap)
    open_file = pyqtSignal(int, int, float, str, str)
    record_frame = pyqtSignal(QImage)
    close_file = pyqtSignal()
    show_fps = pyqtSignal(float)

    @pyqtSlot()
    def run(self):
        try:
            while self.camera.flag_capture:
                start_time = time.time()
                ret, frame = self.camera.cap.read()

                if self.camera.flag_record and not self.flag_open_file:
                    self.open_file.emit(
                        self.camera.width,
                        self.camera.height,
                        self.camera.fps,
                        self.camera.name,
                        self.camera.record_format
                    )

                    self.flag_open_file = True

                if not self.camera.flag_record and self.flag_open_file:
                    self.close_file.emit()
                    self.flag_open_file = False

                if not ret:
                    time.sleep(0.01)
                    continue

                # кадр без коррекции для записи
                raw_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_q_image = QImage(
                    raw_rgb_frame.data,
                    self.camera.width,
                    self.camera.height,
                    self.camera.channel * self.camera.width,
                    QImage.Format.Format_RGB888
                ).copy()

                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Применяем ROI к потоку отображения и обработке.
                # После вырезания масштабируем обратно до исходного размера кадра.
                roi_x = max(0, min(self.camera.roi_x, self.camera.width - 1))
                roi_y = max(0, min(self.camera.roi_y, self.camera.height - 1))
                roi_w = max(1, min(self.camera.roi_width, self.camera.width - roi_x))
                roi_h = max(1, min(self.camera.roi_height, self.camera.height - roi_y))
                roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                if roi_frame.size != 0:
                    frame = cv2.resize(roi_frame, (self.camera.width, self.camera.height), interpolation=cv2.INTER_LINEAR)

                frame = self.processor.process(
                    frame,
                    self.camera.width,
                    self.camera.height,
                    self.camera.roi_x,
                    self.camera.roi_y,
                    self.camera.roi_width,
                    self.camera.roi_height,
                    self.camera.method_for_contrast,
                    self.camera.method_for_noise,
                )

                bytes_per_line = self.camera.channel * self.camera.width

                q_image = QImage(frame.data, self.camera.width, self.camera.height, bytes_per_line, QImage.Format.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)

                self.paint_frame.emit(pixmap)

                if self.camera.flag_record and self.flag_open_file:
                    self.record_frame.emit(raw_q_image)

                end_time = time.time()
                self.current_frame += 1
                self.current_time += (end_time - start_time)
                if self.current_time >= 1.0:
                    self.show_fps.emit(self.current_frame / self.current_time)
                    self.current_time = 0
                    self.current_frame = 0

        except Exception as e:
            print("При захвате кадра с камеры произошла ошибка: ", e)
        finally:
            if self.flag_open_file:
                self.close_file.emit()
                self.flag_open_file = False
            self.camera.disconnect()
