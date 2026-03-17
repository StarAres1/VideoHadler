from PyQt5.QtWidgets import QSpinBox, QSlider
from PyQt5.QtCore import pyqtSlot

class SpinBox_Slider:
    def __init__(self, slider, spinbox, func, slider_value=0, spinbox_value=0, func_slider=None,
                 func_spinbox=None):
        self.spinbox = spinbox
        self.slider = slider

        self.set(slider_value, spinbox_value)

        self.spinbox.valueChanged.connect(func)

        self.spinbox.valueChanged.connect(lambda value: self.set_value(value, slider, func_slider))
        self.slider.sliderMoved.connect(lambda value: self.set_value(value, spinbox, func_spinbox))

    def set(self, slider_value, spinbox_value):
        self.slider.setValue(slider_value)
        self.spinbox.setValue(spinbox_value)

    @pyqtSlot()
    def set_value(self, value, element, func=None):
        if func is not None:
            value = func(value)
        element.setValue(value)

    @staticmethod
    def pow2_int(value):
        return int(value * 2)

    @staticmethod
    def dec2_float(value):
        return float(value / 2)