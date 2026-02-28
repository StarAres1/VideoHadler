import cv2


class ContrastImprover:

    @staticmethod
    def CLAHE(frame, color=True, clipLimit=4.0, titleGridSizeX=10, titleGridSizeY=10):
        if color:
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_frame)

            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(titleGridSizeX, titleGridSizeY))
            l = clahe.apply(l)

            res = cv2.merge((l, a, b))
            res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

            return res
