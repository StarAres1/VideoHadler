from enum import Enum


class ContrastImprovement(Enum):
    NotImprove = 0
    CLAHE = 1
    adjust_contrast = 2
    HE = 3
    gamma = 4
    autoGamma = 5
    sigmoid = 6
    nn = 7


class NoiseReduction(Enum):
    NotReduction = 0
    Blur = 1
    GaussianBlur = 2
    MedianBlur = 3
    BilateralFilter = 4
    FastGaussian = 5
