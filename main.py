from app.core.CameraManager import CameraManager
from forms.main_window_ui import Ui_MainWindow
import threading
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from app.core.Camera import Camera
from app.core.Enums import ContrastImprovement, NoiseReduction

if __name__ == "__main__":
    app = QApplication(sys.argv)

    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    # поиск доступных камер сразу после включения приложения
    cameraManager = CameraManager(ui.videoFrame, ui)
    cameraManager.find_cameras(ui.c_list_cameras)

    # обновление списка доступных камер
    ui.bt_update_cameras.clicked.connect(lambda: cameraManager.find_cameras(ui.c_list_cameras))

    # начать вывод изображения на монитор
    ui.bt_start_capture.clicked.connect(lambda: cameraManager.cameras[0].start_capture())

    # завершить захват кадров
    ui.bt_stop_capture.clicked.connect(lambda: cameraManager.cameras[0].stop_capture())

    # начать запись в файл
    ui.bt_start_record.clicked.connect(cameraManager.cameras[0].start_record)

    # завершить запись в файл
    ui.bt_stop_record.clicked.connect(cameraManager.cameras[0].stop_record)

    # выбор метода улучшения контраста
    ui.r_CLAHE.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.CLAHE))
    ui.r_NotImprove.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.NotImprove))
    ui.r_Retinex.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.Retinex))
    ui.r_HE.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.HE))
    ui.radioButton_4.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.gamma))
    ui.radioButton_5.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.autoGamma))
    ui.radioButton_6.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.sigmoid))
    ui.radioButton.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_contrast(ContrastImprovement.combined))

    # выбор метода устранения шума
    ui.r_NotReductionNoise.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.NotReduction))
    ui.r_Blur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.Blur))
    ui.r_GaussianBlur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.GaussianBlur))
    ui.r_MedianBlur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.MedianBlur))
    ui.r_BilateralFilter.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.BilateralFilter))
    ui.r_FastNlMeansDenoising.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.FastNlMeansDenoising))


    MainWindow.show()
    sys.exit(app.exec_())
