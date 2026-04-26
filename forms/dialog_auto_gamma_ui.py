from PyQt6 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(560, 140)
        self.main_layout = QtWidgets.QGridLayout(Dialog)
        self.main_layout.setObjectName("main_layout")

        self.label_target = QtWidgets.QLabel(parent=Dialog)
        self.label_target.setObjectName("label_target")
        self.main_layout.addWidget(self.label_target, 0, 0, 1, 1)

        self.slider_target = QtWidgets.QSlider(parent=Dialog)
        self.slider_target.setMinimum(1)
        self.slider_target.setMaximum(254)
        self.slider_target.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_target.setObjectName("slider_target")
        self.main_layout.addWidget(self.slider_target, 0, 1, 1, 1)

        self.spin_target = QtWidgets.QSpinBox(parent=Dialog)
        self.spin_target.setMinimum(1)
        self.spin_target.setMaximum(254)
        self.spin_target.setObjectName("spin_target")
        self.main_layout.addWidget(self.spin_target, 0, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Настройки авто-гаммы"))
        self.label_target.setText(_translate("Dialog", "Целевая яркость"))
