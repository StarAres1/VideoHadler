from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from forms.main_window_ui import Ui_MainWindow
from app.core.SpinBox_Slider import SpinBox_Slider

from forms.dialog_clahe_ui import Ui_Dialog as ClaheWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.camera = None

    @pyqtSlot()
    def show_dialog_CLAHE(self):
        self.dialog_clahe = QtWidgets.QDialog()
        ui = ClaheWindow()
        ui.setupUi(self.dialog_clahe)
        self.dialog_clahe.show()

        self.sl_sp_titleGrid = SpinBox_Slider(ui.slider_titlleGrid, ui.spB_titleGrid, self.camera.set_titleGridSize_CLAHE,
                                         4, 4, None, None)

        self.sl_sp_clipLimit = SpinBox_Slider(ui.slider_clipLimit, ui.spB_clipLimit, self.camera.set_clipLimit_CLAHE,
                                         4, 2.0, SpinBox_Slider.pow2_int, SpinBox_Slider.dec2_float)
