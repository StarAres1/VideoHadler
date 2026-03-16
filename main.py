from app.core.CameraManager import CameraManager
from forms.main_window_ui import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from app.core.Enums import ContrastImprovement, NoiseReduction
from PyQt5.QtCore import QFile, QTextStream
from app.core.MainWindow import *
from app.core.MainWindow import MainWindow

#TODO: добавить в CameraManager метод currentCamera
def load_stylesheet(filename):
    file = QFile(filename)
    if not file.open(QFile.ReadOnly | QFile.Text):
        return ""
    stream = QTextStream(file)
    return stream.readAll()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet("styles/style.qss"))

    mainWindow = MainWindow()
    #ui = Ui_MainWindow()
    #ui.setupUi(MainWindow)

    # поиск доступных камер сразу после включения приложения
    cameraManager = CameraManager(mainWindow.ui.videoFrame, mainWindow.ui)
    cameraManager.find_cameras(mainWindow.ui.c_list_cameras)
    mainWindow.camera = cameraManager.cameras[0]

    # обновление списка доступных камер
    mainWindow.ui.bt_update_cameras.clicked.connect(lambda: cameraManager.find_cameras(mainWindow.ui.c_list_cameras))

    # начать вывод изображения на монитор
    mainWindow.ui.bt_start_capture.clicked.connect(lambda: cameraManager.cameras[0].start_capture())

    # завершить захват кадров
    mainWindow.ui.bt_stop_capture.clicked.connect(lambda: cameraManager.cameras[0].stop_capture())

    # начать запись в файл
    mainWindow.ui.bt_start_record.clicked.connect(cameraManager.cameras[0].start_record)

    # завершить запись в файл
    mainWindow.ui.bt_stop_record.clicked.connect(cameraManager.cameras[0].stop_record)

    # выбор метода улучшения контраста
    mainWindow.ui.r_CLAHE.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.CLAHE))
    mainWindow.ui.r_NotImprove.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.NotImprove))
    mainWindow.ui.r_Retinex.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.Retinex))
    mainWindow.ui.r_HE.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.HE))
    mainWindow.ui.radioButton_4.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.gamma))
    mainWindow.ui.radioButton_5.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.autoGamma))
    mainWindow.ui.radioButton_6.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.sigmoid))
    mainWindow.ui.radioButton.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.combined))

    # выбор метода устранения шума
    mainWindow.ui.r_NotReductionNoise.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.NotReduction))
    mainWindow.ui.r_Blur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.Blur))
    mainWindow.ui.r_GaussianBlur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.GaussianBlur))
    mainWindow.ui.r_MedianBlur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.MedianBlur))
    mainWindow.ui.r_BilateralFilter.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.BilateralFilter))
    mainWindow.ui.r_FastNlMeansDenoising.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.FastNlMeansDenoising))

    mainWindow.ui.bt_CLAHE.clicked.connect(mainWindow.show_dialog_CLAHE)

    mainWindow.show()
    sys.exit(app.exec_())