import cv2
import numpy as np
from skimage.exposure import adjust_gamma, adjust_sigmoid

class ContrastImprover:

    @staticmethod
    def CLAHE(frame, color=True, clipLimit=4.0, titleGridSizeX=8, titleGridSizeY=8):
        if color:
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
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
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_frame)

            l = cv2.equalizeHist(l)

            res = cv2.merge((l, a, b))
            res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

            return res

    @staticmethod
    def Retinex(img_bgr, alpha=2, beta=10):
        enhanced_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        # 2. Конвертируем BGR -> RGB
        img_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb

    @staticmethod
    def gamma_correction(frame, gamma=1.5, color=True):

        if color:

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img_normalized = img_rgb.astype(np.float64) / 255.0
            img_corrected = adjust_gamma(img_normalized, gamma=gamma)

            return (img_corrected * 255).astype(np.uint8)
        else:

            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            img_normalized = gray.astype(np.float64) / 255.0
            img_corrected = adjust_gamma(img_normalized, gamma=gamma)
            return (img_corrected * 255).astype(np.uint8)

    @staticmethod
    def sigmoid_correction(frame, cutoff=0.5, gain=10, color=True):
        if color:
            # Конвертируем BGR в RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Нормализуем в [0, 1] для skimage
            img_normalized = img_rgb / 255.0
            # Применяем сигмоидальную коррекцию
            img_corrected = adjust_sigmoid(img_normalized, cutoff=cutoff, gain=gain)
            return (img_corrected * 255).astype(np.uint8)

    @staticmethod
    def auto_gamma(frame, color=True):

        if color:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)

        target = 128
        gamma = np.log(mean_brightness / 255) / np.log(target / 255)

        gamma = np.clip(gamma, 0.2, 5.0)

        #print(f"Автоматически подобранная гамма: {gamma:.2f}")
        return ContrastImprover.gamma_correction(frame, gamma=gamma, color=color)

    @staticmethod
    def combined_enhancement(frame, clip_limit=4.0, sigmoid_gain=8):

        clahe_result = ContrastImprover.CLAHE(frame, clipLimit=clip_limit)

        gamma_result = ContrastImprover.auto_gamma(clahe_result, color=True)

        final_result = ContrastImprover.sigmoid_correction(gamma_result, gain=sigmoid_gain, color=True)

        return final_result
