from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QMainWindow
from forms.main_window_ui import Ui_MainWindow
from app.core.SpinBox_Slider import SpinBox_Slider

from forms.dialog_clahe_ui import Ui_Dialog as ClaheWindow
from forms.dialog_adjust_contrast_ui import Ui_Dialog as AdjustWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.camera = None



    @pyqtSlot()
    def show_dialog_CLAHE(self):
        self.dialog_clahe = QtWidgets.QDialog()
        self.ui_dialog_clahe = ClaheWindow()
        self.ui_dialog_clahe.setupUi(self.dialog_clahe)

        self.sl_sp_titleGrid = SpinBox_Slider(self.ui_dialog_clahe.slider_titlleGrid, self.ui_dialog_clahe.spB_titleGrid, self.camera.set_titleGridSize_CLAHE,
                                         4, 4, None, None)

        self.sl_sp_clipLimit = SpinBox_Slider(self.ui_dialog_clahe.slider_clipLimit, self.ui_dialog_clahe.spB_clipLimit, self.camera.set_clipLimit_CLAHE,
                                         4, 2.0, SpinBox_Slider.pow2_int, SpinBox_Slider.dec2_float)

        self.dialog_clahe.show()

    @pyqtSlot()
    def show_dialog_adjustContrast(self):
        self.dialog_adjust = QtWidgets.QDialog()
        self.ui_dialog_adjust = AdjustWindow()
        self.ui_dialog_adjust.setupUi(self.dialog_adjust)

        self.sl_sp_contrast = SpinBox_Slider(self.ui_dialog_adjust.slisder_contrast, self.ui_dialog_adjust.spB_contrast, self.camera.set_alpha_adjust,
                                         10, 1.0, SpinBox_Slider.pow10_int, SpinBox_Slider.dec10_float)

        self.sl_sp_brightness = SpinBox_Slider(self.ui_dialog_adjust.slider_brightness, self.ui_dialog_adjust.spB_brightness, self.camera.set_beta_adjust,
                                         0, 0, None, None)

        self.dialog_adjust.show()


