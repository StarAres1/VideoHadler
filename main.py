from app.core.CameraManager import CameraManager
import sys
from PyQt6.QtWidgets import QApplication
from app.core.Enums import ContrastImprovement, NoiseReduction
from PyQt6.QtCore import QFile, QTextStream
from app.core.MainWindow import *
from app.core.MainWindow import MainWindow
from app.core.FileBrowser import FileBrowser
from app.core.VideoPlayer import VideoPlayer


def load_stylesheet(filename):
    file = QFile(filename)
    if not file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        return ""
    stream = QTextStream(file)
    return stream.readAll()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet("styles/style.qss"))

    mainWindow = MainWindow()

    # поиск доступных камер сразу после включения приложения
    cameraManager = CameraManager(mainWindow.ui.videoFrame, mainWindow.ui)
    cameraManager.find_cameras(mainWindow.ui.c_list_cameras)
    mainWindow.camera = cameraManager.current_camera(0)
    if mainWindow.camera is None:
        mainWindow.statusBar().showMessage("Камеры не найдены. Доступен только режим воспроизведения файла.", 5000)

    # обновление списка доступных камер
    mainWindow.ui.bt_update_cameras.clicked.connect(lambda: cameraManager.find_cameras(mainWindow.ui.c_list_cameras))

    def apply_to_active_sources(method_name, value):
        if mainWindow.camera and getattr(mainWindow.camera, "flag_capture", False) and hasattr(mainWindow.camera, method_name):
            if value is None:
                getattr(mainWindow.camera, method_name)()
            else:
                getattr(mainWindow.camera, method_name)(value)
        if mainWindow.videoPlayer and mainWindow.videoPlayer.is_loaded() and hasattr(mainWindow.videoPlayer, method_name):
            if value is None:
                getattr(mainWindow.videoPlayer, method_name)()
            else:
                getattr(mainWindow.videoPlayer, method_name)(value)

    # переключение активной камеры из выпадающего списка
    previous_camera_index = [mainWindow.ui.c_list_cameras.currentIndex()]

    def stop_camera_pipeline():
        if mainWindow.camera and mainWindow.camera.flag_record:
            mainWindow.stop_camera_recording()
        if mainWindow.camera and mainWindow.camera.flag_capture:
            mainWindow.camera.stop_capture()
        mainWindow.set_camera_stream_active(False)

    def stop_file_playback():
        if mainWindow.videoPlayer and mainWindow.videoPlayer.is_loaded():
            mainWindow.videoPlayer.stop()
        mainWindow.ui.groupBox_2.setVisible(False)
        mainWindow.ui.label_11.setText("0:00:00")
        if not (mainWindow.camera and mainWindow.camera.flag_capture):
            mainWindow.set_camera_stream_active(False)

    def on_camera_changed(index):
        if index < 0:
            return
        if mainWindow.videoPlayer and mainWindow.videoPlayer.is_loaded():
            approved = mainWindow.ask_user_confirmation(
                "Переключение на камеру",
                "Сейчас идет воспроизведение видеофайла. Оно будет остановлено. Продолжить?"
            )
            if not approved:
                mainWindow.ui.c_list_cameras.blockSignals(True)
                mainWindow.ui.c_list_cameras.setCurrentIndex(previous_camera_index[0])
                mainWindow.ui.c_list_cameras.blockSignals(False)
                return
            stop_file_playback()

        if mainWindow.camera and mainWindow.camera.flag_capture and index != previous_camera_index[0]:
            approved = mainWindow.ask_user_confirmation(
                "Смена камеры",
                "Текущий захват с камеры будет остановлен, а запись в файл (если идет) прервана. Продолжить?"
            )
            if not approved:
                mainWindow.ui.c_list_cameras.blockSignals(True)
                mainWindow.ui.c_list_cameras.setCurrentIndex(previous_camera_index[0])
                mainWindow.ui.c_list_cameras.blockSignals(False)
                return
            stop_camera_pipeline()

        selected = cameraManager.current_camera(index)
        if selected:
            mainWindow.camera = selected
            init_roi_controls()
            previous_camera_index[0] = index

    mainWindow.ui.c_list_cameras.currentIndexChanged.connect(on_camera_changed)

    # ROI controls
    def init_roi_controls():
        cam = mainWindow.camera
        if not cam:
            return

        frame_w = cam.width if cam.width else 640
        frame_h = cam.height if cam.height else 480

        controls = [
            (mainWindow.ui.horizontalSlider, mainWindow.ui.spinBox, frame_w - 1),      # X
            (mainWindow.ui.horizontalSlider_2, mainWindow.ui.spinBox_2, frame_h - 1),  # Y
            (mainWindow.ui.horizontalSlider_3, mainWindow.ui.spinBox_3, frame_w),      # ROI width
            (mainWindow.ui.horizontalSlider_4, mainWindow.ui.spinBox_4, frame_h),      # ROI height
        ]
        for slider, spin, max_val in controls:
            slider.setMinimum(0 if max_val > 1 else 1)
            spin.setMinimum(0 if max_val > 1 else 1)
            slider.setMaximum(max_val)
            spin.setMaximum(max_val)

        if cam.width and cam.height:
            mainWindow.ui.horizontalSlider_3.setMinimum(1)
            mainWindow.ui.spinBox_3.setMinimum(1)
            mainWindow.ui.horizontalSlider_4.setMinimum(1)
            mainWindow.ui.spinBox_4.setMinimum(1)

            cam.set_roi_x(0)
            cam.set_roi_y(0)
            cam.set_roi_width(cam.width)
            cam.set_roi_height(cam.height)

            mainWindow.ui.spinBox.setValue(0)
            mainWindow.ui.spinBox_2.setValue(0)
            mainWindow.ui.spinBox_3.setValue(cam.width)
            mainWindow.ui.spinBox_4.setValue(cam.height)

        update_roi_limits()

    def update_roi_limits():
        cam = mainWindow.camera
        if not cam and not (mainWindow.videoPlayer and mainWindow.videoPlayer.is_loaded()):
            return
        frame_w = cam.width if cam and cam.width else (mainWindow.videoPlayer.width if mainWindow.videoPlayer else 640)
        frame_h = cam.height if cam and cam.height else (mainWindow.videoPlayer.height if mainWindow.videoPlayer else 480)

        x = mainWindow.ui.spinBox.value()
        y = mainWindow.ui.spinBox_2.value()

        max_x = max(0, frame_w - 1)
        max_y = max(0, frame_h - 1)

        mainWindow.ui.horizontalSlider.setMaximum(max_x)
        mainWindow.ui.spinBox.setMaximum(max_x)
        mainWindow.ui.horizontalSlider_2.setMaximum(max_y)
        mainWindow.ui.spinBox_2.setMaximum(max_y)

        if x > max_x:
            mainWindow.ui.spinBox.setValue(max_x)
            x = max_x
        if y > max_y:
            mainWindow.ui.spinBox_2.setValue(max_y)
            y = max_y

        max_w = max(1, frame_w - x)
        max_h = max(1, frame_h - y)

        mainWindow.ui.horizontalSlider_3.setMinimum(1)
        mainWindow.ui.spinBox_3.setMinimum(1)
        mainWindow.ui.horizontalSlider_4.setMinimum(1)
        mainWindow.ui.spinBox_4.setMinimum(1)

        mainWindow.ui.horizontalSlider_3.setMaximum(max_w)
        mainWindow.ui.spinBox_3.setMaximum(max_w)
        mainWindow.ui.horizontalSlider_4.setMaximum(max_h)
        mainWindow.ui.spinBox_4.setMaximum(max_h)

        if mainWindow.ui.spinBox_3.value() > max_w:
            mainWindow.ui.spinBox_3.setValue(max_w)
        if mainWindow.ui.spinBox_4.value() > max_h:
            mainWindow.ui.spinBox_4.setValue(max_h)

    mainWindow.sl_sp_roi_x = SpinBox_Slider(mainWindow.ui.horizontalSlider, mainWindow.ui.spinBox,
                                            lambda v: apply_to_active_sources("set_roi_x", v),
                                            0, 0)
    mainWindow.sl_sp_roi_y = SpinBox_Slider(mainWindow.ui.horizontalSlider_2, mainWindow.ui.spinBox_2,
                                            lambda v: apply_to_active_sources("set_roi_y", v),
                                            0, 0)
    mainWindow.sl_sp_roi_w = SpinBox_Slider(mainWindow.ui.horizontalSlider_3, mainWindow.ui.spinBox_3,
                                            lambda v: apply_to_active_sources("set_roi_width", v),
                                            1, 1)
    mainWindow.sl_sp_roi_h = SpinBox_Slider(mainWindow.ui.horizontalSlider_4, mainWindow.ui.spinBox_4,
                                            lambda v: apply_to_active_sources("set_roi_height", v),
                                            1, 1)
    mainWindow.set_roi_controls(mainWindow.ui.spinBox, mainWindow.ui.spinBox_2, mainWindow.ui.spinBox_3, mainWindow.ui.spinBox_4)
    mainWindow.set_roi_change_callback(update_roi_limits)

    mainWindow.ui.spinBox.valueChanged.connect(lambda _: update_roi_limits())
    mainWindow.ui.spinBox_2.valueChanged.connect(lambda _: update_roi_limits())
    mainWindow.ui.spinBox_3.valueChanged.connect(lambda _: update_roi_limits())
    mainWindow.ui.spinBox_4.valueChanged.connect(lambda _: update_roi_limits())

    init_roi_controls()

    # единая кнопка старт/стоп захвата
    def toggle_capture():
        if not mainWindow.camera:
            return
        if mainWindow.camera.flag_capture:
            stop_camera_pipeline()
            return

        if mainWindow.videoPlayer and mainWindow.videoPlayer.is_loaded():
            approved = mainWindow.ask_user_confirmation(
                "Переход в режим камеры",
                "Воспроизведение видеофайла будет остановлено. Продолжить?"
            )
            if not approved:
                return
            stop_file_playback()

        mainWindow.camera.start_capture()
        init_roi_controls()
        mainWindow.set_camera_stream_active(True)

    mainWindow.ui.bt_start_capture.clicked.connect(toggle_capture)

    # единая кнопка начать/завершить запись в файл
    mainWindow.ui.bt_start_record.clicked.connect(mainWindow.toggle_camera_recording)

    # выбор метода улучшения контраста
    mainWindow.ui.r_CLAHE.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.CLAHE))
    mainWindow.ui.r_NotImprove.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.NotImprove))
    mainWindow.ui.r_adjust.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.adjust_contrast))
    def on_he_method_selected():
        apply_to_active_sources("set_method_for_contrast", ContrastImprovement.HE)

    mainWindow.ui.r_HE.clicked.connect(on_he_method_selected)
    mainWindow.ui.r_gamma.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.gamma))
    mainWindow.ui.r_sigmoid.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.sigmoid))
    mainWindow.ui.r_auto.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.nn))
    mainWindow.ui.r_auto.clicked.connect(mainWindow.ensure_nn_model_loaded_async)
    mainWindow.ui.radioButton.clicked.connect(lambda: apply_to_active_sources("set_method_for_contrast", ContrastImprovement.autoGamma))

    # формат записи
    mainWindow.ui.c_format.currentTextChanged.connect(
        lambda fmt: mainWindow.camera.set_record_format(fmt) if mainWindow.camera else None
    )

    # выбор метода устранения шума
    mainWindow.ui.r_NotReductionNoise.clicked.connect(lambda: apply_to_active_sources("set_method_for_noise", NoiseReduction.NotReduction))
    #mainWindow.ui.r_Blur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.Blur))
    #mainWindow.ui.r_GaussianBlur.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.GaussianBlur))
    mainWindow.ui.r_MedianBlur.clicked.connect(lambda: apply_to_active_sources("set_method_for_noise", NoiseReduction.MedianBlur))
    #mainWindow.ui.r_BilateralFilter.clicked.connect(lambda: cameraManager.cameras[0].set_method_for_noise(NoiseReduction.BilateralFilter))
    mainWindow.ui.r_FastNlMeansDenoising.clicked.connect(lambda: apply_to_active_sources("set_method_for_noise", NoiseReduction.FastGaussian))

    # вызов диалоговых окон для методов улучшения контраста
    mainWindow.ui.bt_CLAHE.clicked.connect(mainWindow.show_dialog_CLAHE)
    mainWindow.ui.bt_adjust.clicked.connect(mainWindow.show_dialog_adjustContrast)
    mainWindow.ui.bt_gamma.clicked.connect(mainWindow.show_gamma_info)
    mainWindow.ui.bt_sigmoid.clicked.connect(mainWindow.show_sigmoid_info)
    mainWindow.ui.toolButton_4.clicked.connect(mainWindow.show_nn_auto_info)
    mainWindow.ui.bt_auto_gamma.clicked.connect(mainWindow.show_auto_gamma_info)
    mainWindow.ui.toolButton_3.clicked.connect(mainWindow.show_noise_median_info)
    mainWindow.ui.toolButton_8.clicked.connect(mainWindow.show_noise_nlm_info)

    # tree для выбора файла для воспроизведения видео
    mainWindow.tree = FileBrowser(mainWindow.ui.treeView)

    # подготовка VideoPlayer
    mainWindow.videoPlayer = VideoPlayer(mainWindow.ui.videoFrame)
    mainWindow.bind_video_player(mainWindow.videoPlayer)
    def on_video_selected(path):
        if mainWindow.camera and mainWindow.camera.flag_capture:
            approved = mainWindow.ask_user_confirmation(
                "Переход к видеофайлу",
                "Текущий захват с камеры будет остановлен, а запись в файл (если идет) прервана. Продолжить?"
            )
            if not approved:
                return
            stop_camera_pipeline()
        mainWindow.videoPlayer.play(path)

    mainWindow.tree.video_selected.connect(on_video_selected)
    mainWindow.ui.pushButton.clicked.connect(mainWindow.toggle_video_playback)
    mainWindow.ui.pushButton_3.clicked.connect(mainWindow.video_seek_backward)
    mainWindow.ui.pushButton_2.clicked.connect(mainWindow.video_seek_forward)
    mainWindow.ui.pushButton_4.clicked.connect(mainWindow.make_video_screenshot)
    mainWindow.ui.horizontalSlider_5.valueChanged.connect(mainWindow.set_video_position)


    mainWindow.show()
    sys.exit(app.exec())
