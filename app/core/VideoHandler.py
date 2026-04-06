from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import cv2
import time
from app.core.Enums import ContrastImprovement, NoiseReduction
from app.core.ContrastImprover import ContrastImprover
from app.core.QualityImprover import QualityImprover
from app.core.NNContrastSelector import NN_SELECTOR

class VideoHandler(QObject):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.flag_open_file = False

        self.current_frame = 0.0
        self.current_time = 0.0

        # параметры для CLAHE
        self.clipLimit = 2
        self.titleGridSize = 4

        # параметры для adjust
        self.alpha = 1
        self.beta = 0

        # параметры для gamma / sigmoid / auto gamma / HE
        self.gamma = 1.5
        self.sigmoid_cutoff = 0.5
        self.sigmoid_gain = 12.0
        self.auto_gamma_target_brightness = 128
        self.auto_gamma_color = True
        self.he_color = True

        # параметры подавления шума
        self.median_ksize = 3
        self.fast_gaussian_ksize = 3
        self.fast_gaussian_sigma = 1.0
        self.nn_skip_frames = 0
        self._nn_skip_counter = 0
        self._nn_last_label = ""
        self.nn_selector = NN_SELECTOR


    paint_frame = pyqtSignal(QPixmap)
    open_file = pyqtSignal(int, int, float, str, str)
    record_frame = pyqtSignal(QImage)
    close_file = pyqtSignal()
    show_fps = pyqtSignal(float)

    #TODO: Возможно try лучше запихнуть внутрь цикла, т.к. сейчас единственная ошибка захвата приведет к остановке проги, но непонятно как это повлияетс на итоговый fps
    # можно при сильном падении fps попробовать запихнуть цикл в функции и постоянно ее запускать при падении предыддущей - хрень какая-то
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

                    #FIXME: в будущем класс Recorder должен с помощью сигнала сообщать об успехе открытия файла
                    # и начинать отображать таймер записи, аналогично с окончанием записи видео
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

                # секция методов подавления шума
                match self.camera.method_for_noise:
                    case NoiseReduction.NotReduction:
                        pass
                    case NoiseReduction.MedianBlur:
                        frame = QualityImprover.medianBlur(frame, self.median_ksize)
                    case NoiseReduction.FastGaussian:
                        frame = QualityImprover.fast_gaussian(
                            frame,
                            ksize=self.fast_gaussian_ksize,
                            sigma=self.fast_gaussian_sigma
                        )
                    case _:
                        pass

                # секция методов улучшения контраста
                match self.camera.method_for_contrast:
                    case ContrastImprovement.NotImprove:
                        pass
                    case ContrastImprovement.CLAHE:
                        frame = ContrastImprover.CLAHE(frame, clipLimit=self.clipLimit, titleGridSizeX=self.titleGridSize,
                                                       titleGridSizeY=self.titleGridSize)
                    case ContrastImprovement.adjust_contrast:
                        frame = ContrastImprover.adjust_contrast(frame, alpha=self.alpha, beta=self.beta)
                    case ContrastImprovement.HE:
                        frame = ContrastImprover.HE(frame, color=self.he_color)
                    case ContrastImprovement.gamma:
                        frame = ContrastImprover.gamma_correction(frame, gamma=self.gamma)
                    case ContrastImprovement.autoGamma:
                        frame = ContrastImprover.auto_gamma(
                            frame,
                            color=self.auto_gamma_color,
                            target_brightness=self.auto_gamma_target_brightness
                        )
                    case ContrastImprovement.sigmoid:
                        frame = ContrastImprover.sigmoid_correction(
                            frame,
                            cutoff=self.sigmoid_cutoff,
                            gain=self.sigmoid_gain
                        )
                    case ContrastImprovement.nn:
                        if self._nn_skip_counter <= 0 or not self._nn_last_label:
                            predicted = self.nn_selector.predict_label(frame)
                            if predicted:
                                self._nn_last_label = predicted
                            self._nn_skip_counter = max(0, int(self.nn_skip_frames))
                        else:
                            self._nn_skip_counter -= 1
                        if self._nn_last_label:
                            frame = self.nn_selector.apply_label(frame, self._nn_last_label)
                    case _:
                        pass

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
                    #print(f"start {start_time}, end {end_time}, time {self.current_time}, frame {self.current_frame}")
                    self.current_time = 0
                    self.current_frame = 0

        except Exception as e:
            print("При захвате кадра с камеры произошла ошибка: ", e)
        finally:
            if self.flag_open_file:
                self.close_file.emit()
                self.flag_open_file = False
            self.camera.disconnect()
