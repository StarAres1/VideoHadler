from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.ContrastImprover import ContrastImprover
from app.core.QualityImprover import QualityImprover

class VideoHandler(QObject):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.flag_open_file = False

        self.current_frame = 0.0
        self.current_time = 0.0


    paint_frame = pyqtSignal(QPixmap)
    open_file = pyqtSignal(int, int, int, str)
    record_frame = pyqtSignal(QImage)
    close_file = pyqtSignal()
    show_fps = pyqtSignal(float)

    #TODO: Возможно try лучше запихнуть внутрь цикла, т.к. сейчас единственная ошибка захвата приведет к остановке проги, но непонятно как это повлияетс на итоговый fps
    # можно при сильном падении fps попробовать запихнуть цикл в функции и постоянно ее запускать при падении предыддущей - хрень какая-то
    @pyqtSlot()
    def run(self):
        flag_try_close = True
        try:
            while self.camera.flag_capture:
                start_time = time.time()
                ret, frame = self.camera.cap.read()

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

                frame = cv2.flip(frame, 1)

                # секция методов подавления шума
                match self.camera.method_for_noise:
                    case NoiseReduction.NotReduction:
                        pass
                    case NoiseReduction.Blur:
                        frame = QualityImprover.blur(frame)
                    case NoiseReduction.GaussianBlur:
                        frame = QualityImprover.gaussianBlur(frame)
                    case NoiseReduction.MedianBlur:
                        frame = QualityImprover.medianBlur(frame)
                    case NoiseReduction.BilateralFilter:
                        frame = QualityImprover.bilateralFilter(frame)
                    case NoiseReduction.BilateralFilter:
                        frame = QualityImprover.fastNl(frame)
                    case _:
                        pass

                # секция методов улучшения контраста
                match self.camera.method_for_contrast:
                    case ContrastImprovement.NotImprove:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    case ContrastImprovement.CLAHE:
                        frame = ContrastImprover.CLAHE(frame)
                    case ContrastImprovement.Retinex:
                        frame = ContrastImprover.Retinex(frame)
                    case ContrastImprovement.HE:
                        frame = ContrastImprover.HE(frame)
                    case ContrastImprovement.gamma:
                        frame = ContrastImprover.gamma_correction(frame)
                    case ContrastImprovement.autoGamma:
                        frame = ContrastImprover.auto_gamma(frame)
                    case ContrastImprovement.sigmoid:
                        frame = ContrastImprover.sigmoid_correction(frame)
                    case ContrastImprovement.combined:
                        frame = ContrastImprover.combined_enhancement(frame)
                    case _:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                bytes_per_line = self.camera.channel * self.camera.width

                q_image = QImage(frame.data, self.camera.width, self.camera.height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)

                self.paint_frame.emit(pixmap)

                if self.camera.flag_record and self.flag_open_file:
                    self.record_frame.emit(q_image)

                end_time = time.time()
                self.current_frame += 1
                self.current_time += (end_time - start_time)
                if self.current_time >= 1.0:
                    self.show_fps.emit(self.current_frame / self.current_time)
                    #print(f"start {start_time}, end {end_time}, time {self.current_time}, frame {self.current_frame}")
                    self.current_time = 0
                    self.current_frame = 0

        except Exception as e:
            print("При захвате кадра с камеры произошла ошибка: ", e)
        finally:
            self.camera.disconnect()
