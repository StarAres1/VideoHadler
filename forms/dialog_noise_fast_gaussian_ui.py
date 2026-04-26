from PyQt6 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(620, 180)
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

        self.label_sigma = QtWidgets.QLabel(parent=Dialog)
        self.label_sigma.setObjectName("label_sigma")
        self.main_layout.addWidget(self.label_sigma, 1, 0, 1, 1)

        self.slider_sigma = QtWidgets.QSlider(parent=Dialog)
        self.slider_sigma.setMinimum(1)
        self.slider_sigma.setMaximum(30)
        self.slider_sigma.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_sigma.setObjectName("slider_sigma")
        self.main_layout.addWidget(self.slider_sigma, 1, 1, 1, 1)

        self.spin_sigma = QtWidgets.QDoubleSpinBox(parent=Dialog)
        self.spin_sigma.setMinimum(0.1)
        self.spin_sigma.setMaximum(3.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setObjectName("spin_sigma")
        self.main_layout.addWidget(self.spin_sigma, 1, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Настройки быстрого гауссова шумоподавления"))
        self.label_kernel.setText(_translate("Dialog", "Размер ядра"))
        self.label_sigma.setText(_translate("Dialog", "Сигма"))
