from app.core.CameraManager import CameraManager
from forms.main_window_ui import Ui_MainWindow
import threading
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)

    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    # поиск доступных камер сразу после включения приложения
    cameraManager = CameraManager()
    cameraManager.find_cameras(ui.c_list_cameras)

    # обновление списка доступных камер
    ui.bt_update_cameras.clicked.connect(lambda: cameraManager.find_cameras(ui.c_list_cameras))

    # начать вывод изображения на монитор
    ui.bt_start_capture.clicked.connect(lambda: ui.c_list_cameras.currentData().start_capture(ui.videoFrame))

    # завершить захват кадров
    ui.bt_stop_capture.clicked.connect(lambda: ui.c_list_cameras.currentData().stop_capture())

    # начать запись в файл
    ui.bt_start_record.clicked.connect(ui.c_list_cameras.currentData().start_record)

    # завершить запись в файл
    ui.bt_stop_record.clicked.connect(ui.c_list_cameras.currentData().stop_record)

    MainWindow.show()
    sys.exit(app.exec_())
