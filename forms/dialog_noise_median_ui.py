from PyQt6 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(560, 140)
        self.main_layout = QtWidgets.QGridLayout(Dialog)
        self.main_layout.setObjectName("main_layout")

        self.label_kernel = QtWidgets.QLabel(parent=Dialog)
        self.label_kernel.setObjectName("label_kernel")
        self.main_layout.addWidget(self.label_kernel, 0, 0, 1, 1)

        self.slider_kernel = QtWidgets.QSlider(parent=Dialog)
        self.slider_kernel.setMinimum(1)
        self.slider_kernel.setMaximum(30)
        self.slider_kernel.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_kernel.setObjectName("slider_kernel")
        self.main_layout.addWidget(self.slider_kernel, 0, 1, 1, 1)

        self.spin_kernel = QtWidgets.QSpinBox(parent=Dialog)
        self.spin_kernel.setMinimum(3)
        self.spin_kernel.setMaximum(61)
        self.spin_kernel.setObjectName("spin_kernel")
        self.main_layout.addWidget(self.spin_kernel, 0, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Настройки медианного фильтра"))
        self.label_kernel.setText(_translate("Dialog", "Размер ядра"))
