import cv2

class QualityImprover:

    @staticmethod
    def medianBlur(frame, ksize=3):
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(frame, ksize=ksize)

    @staticmethod
    def fast_gaussian(frame, ksize=3, sigma=1.0):
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))
