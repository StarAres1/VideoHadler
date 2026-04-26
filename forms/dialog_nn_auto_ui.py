from PyQt6 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(560, 140)
        self.main_layout = QtWidgets.QGridLayout(Dialog)
        self.main_layout.setObjectName("main_layout")

        self.label_skip = QtWidgets.QLabel(parent=Dialog)
        self.label_skip.setObjectName("label_skip")
        self.main_layout.addWidget(self.label_skip, 0, 0, 1, 1)

        self.slider_skip = QtWidgets.QSlider(parent=Dialog)
        self.slider_skip.setMinimum(0)
        self.slider_skip.setMaximum(60)
        self.slider_skip.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_skip.setObjectName("slider_skip")
        self.main_layout.addWidget(self.slider_skip, 0, 1, 1, 1)

        self.spin_skip = QtWidgets.QSpinBox(parent=Dialog)
        self.spin_skip.setMinimum(0)
        self.spin_skip.setMaximum(60)
        self.spin_skip.setObjectName("spin_skip")
        self.main_layout.addWidget(self.spin_skip, 0, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Настройки автоподбора нейросетью"))
        self.label_skip.setText(_translate("Dialog", "Пропуск кадров после анализа"))
