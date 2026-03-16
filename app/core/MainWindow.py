from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from forms.main_window_ui import Ui_MainWindow

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

        ui.spB_clipLimit.valueChanged.connect(self.camera.set_clipLimit_CLAHE)
        ui.spB_titleGrid.valueChanged.connect(self.camera.set_titleGridSize_CLAHE)