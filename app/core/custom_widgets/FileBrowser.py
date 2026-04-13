from PyQt6.QtWidgets import QTreeView
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtCore import QDir, QStandardPaths, QSortFilterProxyModel, pyqtSignal, QObject
import os

class FileSystemProxyModel(QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row, source_parent):
        source_index = self.sourceModel().index(source_row, 0, source_parent)
        if self.sourceModel().isDir(source_index):
            return True
        file_name = self.sourceModel().fileName(source_index)
        return file_name.lower().endswith(('.avi', '.mp4'))

class FileBrowser(QObject):

    video_selected = pyqtSignal(str)

    def __init__(self, tree: QTreeView) -> None:
        super().__init__()
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())

        self.proxy = FileSystemProxyModel()
        self.proxy.setSourceModel(self.model)

        self.tree = tree
        self.tree.setModel(self.proxy)

        movies_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MoviesLocation)

        if not movies_path or not os.path.exists(movies_path):
            movies_path = QDir.homePath()

        root_index = self.model.index(movies_path)
        self.tree.setRootIndex(self.proxy.mapFromSource(root_index))

        # Скрываем ненужные столбцы
        self.tree.setColumnHidden(1, False)  # размер
        self.tree.setColumnHidden(2, True)  # тип
        self.tree.setColumnHidden(3, False)  # дата
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)


        self.tree.doubleClicked.connect(self.on_double_clicked)

    def on_double_clicked(self, index):
        source_index = self.proxy.mapToSource(index)
        if self.model.isDir(source_index):
            return
        file_path = self.model.filePath(source_index)
        if file_path.lower().endswith(('.avi', '.mp4')):
            self.video_selected.emit(file_path)