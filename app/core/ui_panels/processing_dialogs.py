from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore, QtWidgets

from app.core.Enums import ContrastImprovement

from app.core.custom_widgets.SpinBox_Slider import SpinBox_Slider
from forms.dialog_adjust_contrast_ui import Ui_Dialog as AdjustWindow
from forms.dialog_auto_gamma_ui import Ui_Dialog as AutoGammaWindow
from forms.dialog_clahe_ui import Ui_Dialog as ClaheWindow
from forms.dialog_gamma_ui import Ui_Dialog as GammaWindow
from forms.dialog_nn_auto_ui import Ui_Dialog as NnAutoWindow
from forms.dialog_noise_fast_gaussian_ui import Ui_Dialog as NoiseFastGaussianWindow
from forms.dialog_noise_median_ui import Ui_Dialog as NoiseMedianWindow
from forms.dialog_sigmoid_ui import Ui_Dialog as SigmoidWindow


def _can_show(host) -> bool:
    if host.camera:
        return True
    if host.videoPlayer and host.videoPlayer.is_loaded():
        return True
    host.statusBar().showMessage("Параметры доступны после запуска захвата", 2500)
    return False


def show_dialog_clahe(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    host.dialog_clahe = QtWidgets.QDialog(host)
    host.ui_dialog_clahe = ClaheWindow()
    host.ui_dialog_clahe.setupUi(host.dialog_clahe)
    host.sl_sp_titleGrid = SpinBox_Slider(
        host.ui_dialog_clahe.slider_tile_grid,
        host.ui_dialog_clahe.spin_tile_grid,
        lambda value: apply_callback("set_titleGridSize_CLAHE", value),
        4,
        4,
        None,
        None,
    )
    host.sl_sp_clipLimit = SpinBox_Slider(
        host.ui_dialog_clahe.slider_clip_limit,
        host.ui_dialog_clahe.spin_clip_limit,
        lambda value: apply_callback("set_clipLimit_CLAHE", value),
        4,
        2.0,
        SpinBox_Slider.pow2_int,
        SpinBox_Slider.dec2_float,
    )
    host.dialog_clahe.show()


def show_dialog_adjust_contrast(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    host.dialog_adjust = QtWidgets.QDialog(host)
    host.ui_dialog_adjust = AdjustWindow()
    host.ui_dialog_adjust.setupUi(host.dialog_adjust)
    host.sl_sp_contrast = SpinBox_Slider(
        host.ui_dialog_adjust.slider_contrast,
        host.ui_dialog_adjust.spin_contrast,
        lambda value: apply_callback("set_alpha_adjust", value),
        10,
        1.0,
        SpinBox_Slider.pow10_int,
        SpinBox_Slider.dec10_float,
    )
    host.sl_sp_brightness = SpinBox_Slider(
        host.ui_dialog_adjust.slider_brightness,
        host.ui_dialog_adjust.spin_brightness,
        lambda value: apply_callback("set_beta_adjust", value),
        0,
        0,
        None,
        None,
    )
    host.dialog_adjust.show()


def show_gamma_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    ui = GammaWindow()
    ui.setupUi(dialog)
    host.sl_sp_gamma = SpinBox_Slider(
        ui.slider_gamma,
        ui.spin_gamma,
        lambda value: apply_callback("set_gamma_value", value),
        15,
        1.5,
        SpinBox_Slider.pow10_int,
        SpinBox_Slider.dec10_float,
    )
    dialog.show()
    host.dialog_gamma = dialog


def show_sigmoid_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    ui = SigmoidWindow()
    ui.setupUi(dialog)
    host.sl_sp_sigmoid_cutoff = SpinBox_Slider(
        ui.slider_cutoff,
        ui.spin_cutoff,
        lambda value: apply_callback("set_sigmoid_cutoff", value),
        50,
        0.5,
        lambda v: int(v * 100),
        lambda v: float(v / 100.0),
    )
    host.sl_sp_sigmoid_gain = SpinBox_Slider(
        ui.slider_gain,
        ui.spin_gain,
        lambda value: apply_callback("set_sigmoid_gain", value),
        12,
        12,
    )
    dialog.show()
    host.dialog_sigmoid = dialog


def show_auto_gamma_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    ui = AutoGammaWindow()
    ui.setupUi(dialog)
    host.sl_sp_auto_gamma = SpinBox_Slider(
        ui.slider_target,
        ui.spin_target,
        lambda value: apply_callback("set_auto_gamma_target_brightness", value),
        128,
        128,
    )
    dialog.show()
    host.dialog_auto_gamma = dialog


def show_nn_auto_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    ui = NnAutoWindow()
    ui.setupUi(dialog)
    host.sl_sp_nn_skip = SpinBox_Slider(
        ui.slider_skip,
        ui.spin_skip,
        lambda value: apply_callback("set_nn_skip_frames", value),
        0,
        0,
    )
    dialog.show()
    host.dialog_nn_auto = dialog


def show_zero_dce_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    dialog.setWindowTitle("Параметры Zero-DCE")
    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel("Сила эффекта (0.0 - исходный кадр, 1.0 - полный Zero-DCE)")
    layout.addWidget(label)

    row = QtWidgets.QHBoxLayout()
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, dialog)
    slider.setRange(0, 100)
    spin = QtWidgets.QDoubleSpinBox(dialog)
    spin.setDecimals(2)
    spin.setSingleStep(0.01)
    spin.setRange(0.0, 1.0)
    row.addWidget(slider)
    row.addWidget(spin)
    layout.addLayout(row)

    current_strength = 1.0
    if host.camera and getattr(host.camera, "flag_capture", False) and getattr(host.camera, "video_handler", None):
        current_strength = float(getattr(host.camera.video_handler.processor.config, "zero_dce_strength", 1.0))
    elif host.videoPlayer and host.videoPlayer.is_loaded():
        current_strength = float(getattr(host.videoPlayer.processor.config, "zero_dce_strength", 1.0))
    current_strength = max(0.0, min(1.0, current_strength))

    host.sl_sp_zero_dce_strength = SpinBox_Slider(
        slider,
        spin,
        lambda value: apply_callback("set_zero_dce_strength", value),
        int(round(current_strength * 100)),
        current_strength,
        lambda v: int(v * 100),
        lambda v: float(v / 100.0),
    )
    dialog.show()
    host.dialog_zero_dce = dialog


def show_enlightengan_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    dialog.setWindowTitle("Параметры EnlightenGAN")
    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel("Сила эффекта (0.0 - исходный кадр, 1.0 - полный EnlightenGAN)")
    layout.addWidget(label)

    row = QtWidgets.QHBoxLayout()
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, dialog)
    slider.setRange(0, 100)
    spin = QtWidgets.QDoubleSpinBox(dialog)
    spin.setDecimals(2)
    spin.setSingleStep(0.01)
    spin.setRange(0.0, 1.0)
    row.addWidget(slider)
    row.addWidget(spin)
    layout.addLayout(row)

    current_strength = 1.0
    if host.camera and getattr(host.camera, "flag_capture", False) and getattr(host.camera, "video_handler", None):
        current_strength = float(getattr(host.camera.video_handler.processor.config, "enlightengan_strength", 1.0))
    elif host.videoPlayer and host.videoPlayer.is_loaded():
        current_strength = float(getattr(host.videoPlayer.processor.config, "enlightengan_strength", 1.0))
    current_strength = max(0.0, min(1.0, current_strength))

    host.sl_sp_enlightengan_strength = SpinBox_Slider(
        slider,
        spin,
        lambda value: apply_callback("set_enlightengan_strength", value),
        int(round(current_strength * 100)),
        current_strength,
        lambda v: int(v * 100),
        lambda v: float(v / 100.0),
    )
    dialog.show()
    host.dialog_enlightengan = dialog


def show_noise_median_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    ui = NoiseMedianWindow()
    ui.setupUi(dialog)
    host.sl_sp_median = SpinBox_Slider(
        ui.slider_kernel,
        ui.spin_kernel,
        lambda value: apply_callback("set_median_ksize", value),
        1,
        3,
        lambda v: max(3, v * 2 + 1),
        lambda v: max(1, int((v - 1) / 2)),
    )
    dialog.show()
    host.dialog_median = dialog


def show_noise_fast_gaussian_info(host, apply_callback: Callable[[str, object], None]) -> None:
    if not _can_show(host):
        return
    dialog = QtWidgets.QDialog(host)
    ui = NoiseFastGaussianWindow()
    ui.setupUi(dialog)
    host.sl_sp_fast_gauss_k = SpinBox_Slider(
        ui.slider_kernel,
        ui.spin_kernel,
        lambda value: apply_callback("set_fast_gaussian_ksize", value),
        1,
        3,
        lambda v: max(3, v * 2 + 1),
        lambda v: max(1, int((v - 1) / 2)),
    )
    host.sl_sp_fast_gauss_sigma = SpinBox_Slider(
        ui.slider_sigma,
        ui.spin_sigma,
        lambda value: apply_callback("set_fast_gaussian_sigma", value),
        10,
        1.0,
        SpinBox_Slider.pow10_int,
        SpinBox_Slider.dec10_float,
    )
    dialog.show()
    host.dialog_fast_gaussian = dialog


def show_contrast_pipeline_dialog(host) -> None:
    if host._contrast_pipeline_dialog is not None and host._contrast_pipeline_dialog.isVisible():
        host._contrast_pipeline_dialog.raise_()
        host._contrast_pipeline_dialog.activateWindow()
        return

    options = [
        ("CLAHE", ContrastImprovement.CLAHE),
        ("Линейное преобразование", ContrastImprovement.adjust_contrast),
        ("Эквализация гистограммы (HE)", ContrastImprovement.HE),
        ("Гамма-коррекция", ContrastImprovement.gamma),
        ("Сигмоидная коррекция", ContrastImprovement.sigmoid),
        ("Автогамма", ContrastImprovement.autoGamma),
        ("Автоподбор нейросетью", ContrastImprovement.nn),
        ("Zero-DCE", ContrastImprovement.zero_dce),
        ("EnlightenGAN", ContrastImprovement.enlightengan),
    ]
    title_to_method = {title: method for title, method in options}
    method_to_title = {method: title for title, method in options}

    class _PipelineSelectedListWidget(QtWidgets.QListWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._on_after_drop = None

        def dropEvent(self, event):
            super().dropEvent(event)
            for i in range(self.count()):
                item = self.item(i)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) is None:
                    method = title_to_method.get(item.text())
                    if method is not None:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, method)
            if callable(self._on_after_drop):
                self._on_after_drop()

    dialog = QtWidgets.QDialog(host)
    dialog.setWindowTitle("Цепочка улучшения контраста")
    dialog.resize(680, 440)
    layout = QtWidgets.QVBoxLayout(dialog)
    hint = QtWidgets.QLabel("Перетащите методы справа налево. Порядок в левом списке = порядок применения.")
    layout.addWidget(hint)

    lists_row = QtWidgets.QHBoxLayout()
    selected_box = QtWidgets.QVBoxLayout()
    selected_box.addWidget(QtWidgets.QLabel("Выбранные методы"))
    selected = _PipelineSelectedListWidget(dialog)
    selected.setDragEnabled(True)
    selected.setAcceptDrops(True)
    selected.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
    selected.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
    selected.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    selected.setDropIndicatorShown(True)
    for method in host._contrast_pipeline_methods:
        title = method_to_title.get(method)
        if title is None:
            continue
        it = QtWidgets.QListWidgetItem(title)
        it.setData(QtCore.Qt.ItemDataRole.UserRole, method)
        selected.addItem(it)
    selected_box.addWidget(selected)

    available_box = QtWidgets.QVBoxLayout()
    available_box.addWidget(QtWidgets.QLabel("Доступные методы"))
    available = QtWidgets.QListWidget(dialog)
    available.setDragEnabled(True)
    available.setAcceptDrops(False)
    available.setDefaultDropAction(QtCore.Qt.DropAction.CopyAction)
    available.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
    available.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    for title, method in options:
        it = QtWidgets.QListWidgetItem(title)
        it.setData(QtCore.Qt.ItemDataRole.UserRole, method)
        available.addItem(it)
    available_box.addWidget(available)

    lists_row.addLayout(selected_box, 1)
    lists_row.addLayout(available_box, 1)
    layout.addLayout(lists_row)

    def selected_methods():
        methods = []
        for i in range(selected.count()):
            it = selected.item(i)
            method = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if method is not None:
                methods.append(method)
        return methods

    def apply_pipeline_now():
        methods = selected_methods()
        host._contrast_pipeline_methods = list(methods)
        host._apply_to_active_sources("set_method_for_contrast", ContrastImprovement.pipeline)
        host._apply_to_active_sources("set_contrast_pipeline", methods)

    selected._on_after_drop = apply_pipeline_now

    def add_selected_from_available():
        it = available.currentItem()
        if it is None:
            return
        new_item = QtWidgets.QListWidgetItem(it.text())
        new_item.setData(QtCore.Qt.ItemDataRole.UserRole, it.data(QtCore.Qt.ItemDataRole.UserRole))
        selected.addItem(new_item)
        apply_pipeline_now()

    available.itemDoubleClicked.connect(lambda *_: add_selected_from_available())
    selected.itemDoubleClicked.connect(lambda item: host._configure_pipeline_method(item.data(QtCore.Qt.ItemDataRole.UserRole)))
    selected.model().rowsInserted.connect(lambda *_: apply_pipeline_now())
    selected.model().rowsRemoved.connect(lambda *_: apply_pipeline_now())
    selected.model().rowsMoved.connect(lambda *_: apply_pipeline_now())

    def remove_selected_item():
        row = selected.currentRow()
        if row < 0:
            return
        selected.takeItem(row)
        apply_pipeline_now()

    def on_key_press(ev):
        if ev.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            remove_selected_item()
        else:
            QtWidgets.QListWidget.keyPressEvent(selected, ev)

    selected.keyPressEvent = on_key_press  # type: ignore[method-assign]

    dialog.finished.connect(lambda *_: setattr(host, "_contrast_pipeline_dialog", None))
    host.statusBar().showMessage("Окно цепочки открыто: изменения применяются сразу", 2500)
    host._contrast_pipeline_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
