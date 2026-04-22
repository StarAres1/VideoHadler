from __future__ import annotations

import logging

from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from app.neural_network.zero_dce.enhancer import ZERO_DCE_ENHANCER

logger = logging.getLogger(__name__)


class ZeroDceLoadWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    @pyqtSlot()
    def run(self):
        logger.info("Фоновая загрузка модели Zero-DCE: старт")
        ok = ZERO_DCE_ENHANCER.ensure_loaded_with_progress(lambda v, t: self.progress.emit(v, t))
        logger.info("Фоновая загрузка модели Zero-DCE: завершено ok=%s", ok)
        self.finished.emit(ok, ZERO_DCE_ENHANCER.last_error)


def ensure_zero_dce_loaded_async(host) -> None:
    if ZERO_DCE_ENHANCER.is_loaded() or ZERO_DCE_ENHANCER.is_loading():
        logger.debug("Загрузка Zero-DCE: модель уже загружена или загрузка идёт")
        return
    logger.info("Запуск фонового потока загрузки модели контраста (Zero-DCE)")
    host.zero_dce_progress_dialog = QtWidgets.QDialog(host)
    host.zero_dce_progress_dialog.setWindowTitle("Загрузка Zero-DCE")
    host.zero_dce_progress_dialog.setModal(False)
    layout = QtWidgets.QVBoxLayout(host.zero_dce_progress_dialog)
    host.zero_dce_progress_label = QtWidgets.QLabel("Подготовка...")
    host.zero_dce_progress_bar = QtWidgets.QProgressBar()
    host.zero_dce_progress_bar.setRange(0, 100)
    host.zero_dce_progress_bar.setValue(0)
    layout.addWidget(host.zero_dce_progress_label)
    layout.addWidget(host.zero_dce_progress_bar)
    host.zero_dce_progress_dialog.show()

    host.zero_dce_loader_thread = QThread(host)
    host.zero_dce_loader_worker = ZeroDceLoadWorker()
    host.zero_dce_loader_worker.moveToThread(host.zero_dce_loader_thread)
    host.zero_dce_loader_thread.started.connect(host.zero_dce_loader_worker.run)
    host.zero_dce_loader_worker.progress.connect(lambda value, text: _on_zero_dce_load_progress(host, value, text))
    host.zero_dce_loader_worker.finished.connect(lambda ok, error: _on_zero_dce_load_finished(host, ok, error))
    host.zero_dce_loader_worker.finished.connect(host.zero_dce_loader_thread.quit)
    host.zero_dce_loader_worker.finished.connect(host.zero_dce_loader_worker.deleteLater)
    host.zero_dce_loader_thread.finished.connect(host.zero_dce_loader_thread.deleteLater)
    host.zero_dce_loader_thread.start()


def _on_zero_dce_load_progress(host, value: int, text: str) -> None:
    if host.zero_dce_progress_bar:
        host.zero_dce_progress_bar.setValue(value)
    if host.zero_dce_progress_label:
        host.zero_dce_progress_label.setText(text)


def _on_zero_dce_load_finished(host, ok: bool, error: str) -> None:
    logger.info("Загрузка модели Zero-DCE завершена ok=%s", ok)
    if host.zero_dce_progress_dialog:
        host.zero_dce_progress_dialog.close()
        host.zero_dce_progress_dialog = None
    if not ok:
        logger.error("Ошибка загрузки Zero-DCE: %s", error)
        host.statusBar().showMessage(f"Ошибка загрузки Zero-DCE: {error}", 5000)
