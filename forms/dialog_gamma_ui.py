from PyQt6 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(560, 140)
        self.main_layout = QtWidgets.QGridLayout(Dialog)
        self.main_layout.setObjectName("main_layout")

        self.label_gamma = QtWidgets.QLabel(parent=Dialog)
        self.label_gamma.setObjectName("label_gamma")
        self.main_layout.addWidget(self.label_gamma, 0, 0, 1, 1)

        self.slider_gamma = QtWidgets.QSlider(parent=Dialog)
        self.slider_gamma.setMinimum(2)
        self.slider_gamma.setMaximum(50)
        self.slider_gamma.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_gamma.setObjectName("slider_gamma")
        self.main_layout.addWidget(self.slider_gamma, 0, 1, 1, 1)

        self.spin_gamma = QtWidgets.QDoubleSpinBox(parent=Dialog)
        self.spin_gamma.setMinimum(0.2)
        self.spin_gamma.setMaximum(5.0)
        self.spin_gamma.setSingleStep(0.1)
        self.spin_gamma.setObjectName("spin_gamma")
        self.main_layout.addWidget(self.spin_gamma, 0, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Настройки гамма-коррекции"))
        self.label_gamma.setText(_translate("Dialog", "Гамма"))
