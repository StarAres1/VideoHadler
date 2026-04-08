import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QFile, QTextStream
from app.core.MainWindow import MainWindow
from app.core.AppController import AppController


def load_stylesheet(filename):
    file = QFile(filename)
    if not file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        return ""
    stream = QTextStream(file)
    return stream.readAll()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet("styles/style.qss"))
    main_window = MainWindow()
    AppController(main_window)
    main_window.show()
    sys.exit(app.exec())
