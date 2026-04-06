import cv2
import numpy as np
from skimage.exposure import adjust_gamma, adjust_sigmoid

class ContrastImprover:

    @staticmethod
    def CLAHE(frame, color=True, clipLimit=4.0, titleGridSizeX=8, titleGridSizeY=8):
        if color:
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab_frame)

            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(titleGridSizeX, titleGridSizeY))
            l = clahe.apply(l)

            res = cv2.merge((l, a, b))
            res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

            return res

    # Histogram Equalization
    @staticmethod
    def HE(frame, color=True):
        if color:
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab_frame)

            l = cv2.equalizeHist(l)

            res = cv2.merge((l, a, b))
            res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

            return res

    @staticmethod
    def adjust_contrast(img_bgr, alpha=2, beta=10):
        enhanced_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        return enhanced_bgr

    @staticmethod
    def gamma_correction(frame, gamma=1.5, color=True):
        """
        Применяет гамма-коррекцию только к яркостному каналу в LAB.
        Лучше сохраняет цвета.
        """
        # Конвертируем в LAB
        # Убедитесь, что frame в RGB (как в вашем коде). Если BGR, нужно cv2.COLOR_BGR2LAB.
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        # Разделяем каналы
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Нормализуем L канал (0-255 -> 0.0-1.0) и применяем гамму
        l_normalized = l_channel.astype(np.float64) / 255.0
        l_corrected = adjust_gamma(l_normalized, gamma=gamma)

        # Возвращаем в диапазон 0-255 и целочисленный тип
        l_channel_corrected = (l_corrected * 255).astype(np.uint8)

        # Собираем каналы обратно
        lab_corrected = cv2.merge([l_channel_corrected, a_channel, b_channel])

        # Конвертируем обратно в RGB
        rgb_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

        return rgb_corrected
    @staticmethod
    def sigmoid_correction(frame, cutoff=0.5, gain=12, color=True):
        if color:
            # Конвертируем BGR в RGB
            img_rgb = frame
            # Нормализуем в [0, 1] для skimage
            img_normalized = img_rgb / 255.0
            # Применяем сигмоидальную коррекцию
            img_corrected = adjust_sigmoid(img_normalized, cutoff=cutoff, gain=gain)
            return (img_corrected * 255).astype(np.uint8)

    @staticmethod
    def auto_gamma(frame, color=True, target_brightness=128):
        """
        Автоматически подбирает гамму и применяет коррекцию.
        Для цветных изображений используется LAB, для ч/б – напрямую яркость.
        """
        if color:
            # Конвертируем в LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Используем медиану канала яркости для устойчивости
            current_l = np.median(l_channel)

            # Избегаем деления на ноль и крайних значений
            if current_l == 0:
                current_l = 1
            if current_l == 255:
                current_l = 254

            # Расчёт гаммы: хотим привести медиану к target_brightness
            # gamma = log( current/255 ) / log( target/255 )
            gamma = np.log(target_brightness / 255.0) / np.log(current_l / 255.0)
            gamma = np.clip(gamma, 0.2, 5.0)  # ограничиваем разумный диапазон

            # Применяем гамму только к L-каналу
            l_norm = l_channel.astype(np.float64) / 255.0
            l_corrected = adjust_gamma(l_norm, gamma=gamma)
            l_corrected = (l_corrected * 255).astype(np.uint8)

            # Собираем обратно
            lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
            result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
            return result

        else:
            # Для ч/б изображений
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            current = np.median(gray)
            if current == 0:
                current = 1
            if current == 255:
                current = 254

            gamma = np.log(target_brightness / 255.0) / np.log(current / 255.0)
            gamma = np.clip(gamma, 0.2, 5.0)

            # Нормализация и коррекция
            img_norm = gray.astype(np.float64) / 255.0
            img_corrected = adjust_gamma(img_norm, gamma=gamma)
            return (img_corrected * 255).astype(np.uint8)

