import cv2
import numpy as np
from skimage.exposure import adjust_gamma, adjust_sigmoid

class ContrastImprover:

    @staticmethod
    def CLAHE(frame, clipLimit=4.0, titleGridSizeX=8, titleGridSizeY=8):
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_frame)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(titleGridSizeX, titleGridSizeY))
        l = clahe.apply(l)
        res = cv2.merge((l, a, b))
        return cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

    # Histogram Equalization
    @staticmethod
    def HE(frame):
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_frame)
        l = cv2.equalizeHist(l)
        res = cv2.merge((l, a, b))
        return cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

    @staticmethod
    def adjust_contrast(img_bgr, alpha=2, beta=10):
        enhanced_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
        return enhanced_bgr

    @staticmethod
    def gamma_correction(frame, gamma=1.5):

        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_normalized = l_channel.astype(np.float64) / 255.0
        l_corrected = adjust_gamma(l_normalized, gamma=gamma)
        l_channel_corrected = (l_corrected * 255).astype(np.uint8)
        lab_corrected = cv2.merge([l_channel_corrected, a_channel, b_channel])
        rgb_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

        return rgb_corrected
    @staticmethod
    def sigmoid_correction(frame, cutoff=0.5, gain=12):
        img_rgb = frame
        img_normalized = img_rgb / 255.0
        img_corrected = adjust_sigmoid(img_normalized, cutoff=cutoff, gain=gain)
        return (img_corrected * 255).astype(np.uint8)

    @staticmethod
    def auto_gamma(frame, target_brightness=128):
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        current_l = np.median(l_channel)
        if current_l == 0:
            current_l = 1
        if current_l == 255:
            current_l = 254
        gamma = np.log(target_brightness / 255.0) / np.log(current_l / 255.0)
        gamma = np.clip(gamma, 0.2, 5.0)
        l_norm = l_channel.astype(np.float64) / 255.0
        l_corrected = adjust_gamma(l_norm, gamma=gamma)
        l_corrected = (l_corrected * 255).astype(np.uint8)
        lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)


