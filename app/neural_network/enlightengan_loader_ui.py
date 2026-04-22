from __future__ import annotations

import logging

from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from app.neural_network.enlightengan.enhancer import ENLIGHTENGAN_ENHANCER

logger = logging.getLogger(__name__)


class EnlightenGanLoadWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    @pyqtSlot()
    def run(self):
        logger.info("Фоновая загрузка модели EnlightenGAN: старт")
        ok = ENLIGHTENGAN_ENHANCER.ensure_loaded_with_progress(lambda v, t: self.progress.emit(v, t))
        logger.info("Фоновая загрузка модели EnlightenGAN: завершено ok=%s", ok)
        self.finished.emit(ok, ENLIGHTENGAN_ENHANCER.last_error)


def ensure_enlightengan_loaded_async(host) -> None:
    if ENLIGHTENGAN_ENHANCER.is_loaded() or ENLIGHTENGAN_ENHANCER.is_loading():
        logger.debug("Загрузка EnlightenGAN: модель уже загружена или загрузка идёт")
        return
    logger.info("Запуск фонового потока загрузки модели контраста (EnlightenGAN)")
    host.enlightengan_progress_dialog = QtWidgets.QDialog(host)
    host.enlightengan_progress_dialog.setWindowTitle("Загрузка EnlightenGAN")
    host.enlightengan_progress_dialog.setModal(False)
    layout = QtWidgets.QVBoxLayout(host.enlightengan_progress_dialog)
    host.enlightengan_progress_label = QtWidgets.QLabel("Подготовка...")
    host.enlightengan_progress_bar = QtWidgets.QProgressBar()
    host.enlightengan_progress_bar.setRange(0, 100)
    host.enlightengan_progress_bar.setValue(0)
    layout.addWidget(host.enlightengan_progress_label)
    layout.addWidget(host.enlightengan_progress_bar)
    host.enlightengan_progress_dialog.show()

    host.enlightengan_loader_thread = QThread(host)
    host.enlightengan_loader_worker = EnlightenGanLoadWorker()
    host.enlightengan_loader_worker.moveToThread(host.enlightengan_loader_thread)
    host.enlightengan_loader_thread.started.connect(host.enlightengan_loader_worker.run)
    host.enlightengan_loader_worker.progress.connect(
        lambda value, text: _on_enlightengan_load_progress(host, value, text)
    )
    host.enlightengan_loader_worker.finished.connect(
        lambda ok, error: _on_enlightengan_load_finished(host, ok, error)
    )
    host.enlightengan_loader_worker.finished.connect(host.enlightengan_loader_thread.quit)
    host.enlightengan_loader_worker.finished.connect(host.enlightengan_loader_worker.deleteLater)
    host.enlightengan_loader_thread.finished.connect(host.enlightengan_loader_thread.deleteLater)
    host.enlightengan_loader_thread.start()


def _on_enlightengan_load_progress(host, value: int, text: str) -> None:
    if host.enlightengan_progress_bar:
        host.enlightengan_progress_bar.setValue(value)
    if host.enlightengan_progress_label:
        host.enlightengan_progress_label.setText(text)


def _on_enlightengan_load_finished(host, ok: bool, error: str) -> None:
    logger.info("Загрузка модели EnlightenGAN завершена ok=%s", ok)
    if host.enlightengan_progress_dialog:
        host.enlightengan_progress_dialog.close()
        host.enlightengan_progress_dialog = None
    if not ok:
        logger.error("Ошибка загрузки EnlightenGAN: %s", error)
        host.statusBar().showMessage(f"Ошибка загрузки EnlightenGAN: {error}", 5000)
