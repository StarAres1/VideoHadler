from PyQt6 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(620, 180)
        self.main_layout = QtWidgets.QGridLayout(Dialog)
        self.main_layout.setObjectName("main_layout")

        self.label_cutoff = QtWidgets.QLabel(parent=Dialog)
        self.label_cutoff.setObjectName("label_cutoff")
        self.main_layout.addWidget(self.label_cutoff, 0, 0, 1, 1)

        self.slider_cutoff = QtWidgets.QSlider(parent=Dialog)
        self.slider_cutoff.setMinimum(1)
        self.slider_cutoff.setMaximum(99)
        self.slider_cutoff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_cutoff.setObjectName("slider_cutoff")
        self.main_layout.addWidget(self.slider_cutoff, 0, 1, 1, 1)

        self.spin_cutoff = QtWidgets.QDoubleSpinBox(parent=Dialog)
        self.spin_cutoff.setMinimum(0.01)
        self.spin_cutoff.setMaximum(0.99)
        self.spin_cutoff.setSingleStep(0.01)
        self.spin_cutoff.setObjectName("spin_cutoff")
        self.main_layout.addWidget(self.spin_cutoff, 0, 2, 1, 1)

        self.label_gain = QtWidgets.QLabel(parent=Dialog)
        self.label_gain.setObjectName("label_gain")
        self.main_layout.addWidget(self.label_gain, 1, 0, 1, 1)

        self.slider_gain = QtWidgets.QSlider(parent=Dialog)
        self.slider_gain.setMinimum(1)
        self.slider_gain.setMaximum(30)
        self.slider_gain.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_gain.setObjectName("slider_gain")
        self.main_layout.addWidget(self.slider_gain, 1, 1, 1, 1)

        self.spin_gain = QtWidgets.QSpinBox(parent=Dialog)
        self.spin_gain.setMinimum(1)
        self.spin_gain.setMaximum(30)
        self.spin_gain.setObjectName("spin_gain")
        self.main_layout.addWidget(self.spin_gain, 1, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Настройки сигмоидной коррекции"))
        self.label_cutoff.setText(_translate("Dialog", "Отсечка"))
        self.label_gain.setText(_translate("Dialog", "Коэффициент"))
