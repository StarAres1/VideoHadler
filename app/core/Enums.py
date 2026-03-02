from enum import Enum


class ContrastImprovement(Enum):
    NotImprove = 0
    CLAHE = 1
    Retinex = 2
    HE = 3
