import cv2
import numpy as np

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


    def Retinex(img_bgr, alpha=2, beta=10):
        enhanced_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        # 2. Конвертируем BGR -> RGB
        img_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb
