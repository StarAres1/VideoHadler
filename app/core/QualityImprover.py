import cv2
import numpy as np


class QualityImprover:

    @staticmethod
    def blur(frame):
        return cv2.blur(frame, ksize=(3, 3))

    @staticmethod
    def gaussianBlur(frame):
        return cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=2)

    @staticmethod
    def medianBlur(frame):
        return cv2.medianBlur(frame, ksize=3)

    @staticmethod
    def bilateralFilter(frame):
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    @staticmethod
    def fastNl(frame):
        result = cv2.fastNlMeansDenoisingColored(
            frame,
            h=30,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return result
