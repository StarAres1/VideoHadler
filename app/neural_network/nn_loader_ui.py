from __future__ import annotations

import logging

from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from app.neural_network.NNContrastSelector import NN_SELECTOR

logger = logging.getLogger(__name__)


class NNModelLoadWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    @pyqtSlot()
    def run(self):
        logger.info("Фоновая загрузка модели NNContrastSelector: старт")
        ok = NN_SELECTOR.ensure_loaded_with_progress(lambda v, t: self.progress.emit(v, t))
        logger.info("Фоновая загрузка модели NNContrastSelector: завершено ok=%s", ok)
        self.finished.emit(ok, NN_SELECTOR.last_error)


def ensure_nn_model_loaded_async(host) -> None:
    if NN_SELECTOR.is_loaded() or NN_SELECTOR.is_loading():
        logger.debug("Загрузка NN: модель уже загружена или загрузка идёт")
        return
    logger.info("Запуск фонового потока загрузки модели контраста (NN)")
    host.nn_progress_dialog = QtWidgets.QDialog(host)
    host.nn_progress_dialog.setWindowTitle("Загрузка нейросети")
    host.nn_progress_dialog.setModal(False)
    layout = QtWidgets.QVBoxLayout(host.nn_progress_dialog)
    host.nn_progress_label = QtWidgets.QLabel("Подготовка...")
    host.nn_progress_bar = QtWidgets.QProgressBar()
    host.nn_progress_bar.setRange(0, 100)
    host.nn_progress_bar.setValue(0)
    layout.addWidget(host.nn_progress_label)
    layout.addWidget(host.nn_progress_bar)
    host.nn_progress_dialog.show()

    host.nn_loader_thread = QThread(host)
    host.nn_loader_worker = NNModelLoadWorker()
    host.nn_loader_worker.moveToThread(host.nn_loader_thread)
    host.nn_loader_thread.started.connect(host.nn_loader_worker.run)
    host.nn_loader_worker.progress.connect(lambda value, text: _on_nn_load_progress(host, value, text))
    host.nn_loader_worker.finished.connect(lambda ok, error: _on_nn_load_finished(host, ok, error))
    host.nn_loader_worker.finished.connect(host.nn_loader_thread.quit)
    host.nn_loader_worker.finished.connect(host.nn_loader_worker.deleteLater)
    host.nn_loader_thread.finished.connect(host.nn_loader_thread.deleteLater)
    host.nn_loader_thread.start()


def _on_nn_load_progress(host, value: int, text: str) -> None:
    if host.nn_progress_bar:
        host.nn_progress_bar.setValue(value)
    if host.nn_progress_label:
        host.nn_progress_label.setText(text)


def _on_nn_load_finished(host, ok: bool, error: str) -> None:
    logger.info("Загрузка модели контраста завершена ok=%s", ok)
    if host.nn_progress_dialog:
        host.nn_progress_dialog.close()
        host.nn_progress_dialog = None
    if not ok:
        logger.error("Ошибка загрузки нейросети: %s", error)
        host.statusBar().showMessage(f"Ошибка загрузки нейросети: {error}", 5000)
