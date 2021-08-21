import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore,QtGui

import os
import shutil


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class WINDOW(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cnt = 0

    def initUI(self):
        self.setGeometry(400, 200, 1280, 720)
        self.setWindowTitle('Image Retrieval')
        # self.setWindowIcon(QIcon('icon.jpg'))

        QToolTip.setFont(QFont('Times New Roman', 14))

        self.present = QLabel()
        uploadbtn = QPushButton('  Upload an image  ')
        uploadbtn.setToolTip('Open an image file for retrieval')
        uploadbtn.setStyleSheet(_fromUtf8("QPushButton{\n"
                                        "font-size:26px;  \n"
                                        "color:#f2f2f2;  \n"
                                        "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                        "background-color: rgba(0,33,99, 255);\n"
                                        "border:2px solid white;border-radius:10px;\n"
                                        "}\n"
                                        " \n"
                                        "QPushButton:hover{ \n"
                                        "font-size:26px;  \n"
                                        "color:#f2f2f2;  \n"
                                        "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                        "background-color: rgba(0,33,99, 150);\n"
                                        "border:2px solid white;border-radius:10px;\n"
                                        "}\n"
                                        " \n"
                                        "QPushButton:pressed{\n"
                                        "font-size:26px;  \n"
                                        "color:#f2f2f2;  \n"
                                        "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                        "background-color: rgba(0,33,99, 255);\n"
                                        "border:2px solid white;border-radius:10px;\n"
                                        "padding-left:3px;\n"
                                        "padding-top:3px;\n"
                                        "} "))
        uploadbtn.setCheckable(True)
        uploadbtn.setAutoExclusive(True)
        uploadbtn.clicked.connect(self.uploadinputimage)

        clearbtn = QPushButton('  Clear all images  ')
        clearbtn.setToolTip('Delete the image uploaded.')
        clearbtn.setStyleSheet(_fromUtf8("QPushButton{\n"
                                        "font-size:26px;  \n"
                                        "color:#f2f2f2;  \n"
                                        "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                        "background-color: rgba(0,33,99, 255);\n"
                                        "border:2px solid white;border-radius:10px;\n"
                                        "}\n"
                                        " \n"
                                        "QPushButton:hover{ \n"
                                        "font-size:26px;  \n"
                                        "color:#f2f2f2;  \n"
                                        "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                        "background-color: rgba(0,33,99, 150);\n"
                                        "border:2px solid white;border-radius:10px;\n"
                                        "}\n"
                                        " \n"
                                        "QPushButton:pressed{\n"
                                        "font-size:26px;  \n"
                                        "color:#f2f2f2;  \n"
                                        "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                        "background-color: rgba(0,33,99, 255);\n"
                                        "border:2px solid white;border-radius:10px;\n"
                                        "padding-left:3px;\n"
                                        "padding-top:3px;\n"
                                        "} "))
        clearbtn.setCheckable(True)
        clearbtn.setAutoExclusive(True)
        clearbtn.clicked.connect(self.clearinputimage)

        runbtn = QPushButton('  Search  ')
        runbtn.setToolTip('Run the model')
        runbtn.setStyleSheet(_fromUtf8("QPushButton{\n"
                                       "font-size:26px;  \n"
                                       "color:#f2f2f2;  \n"
                                       "font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                       "background-color: rgba(95, 255, 37,255);\n"
                                       "border:2px solid white;border-radius:10px;\n"
                                       "}\n"
                                       "QPushButton:hover{\n"
                                       " font-size:26px;  \n"
                                       "color:#f2f2f2;  \n"
                                       " font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                       "background-color: rgba(95, 255, 37, 180);\n"
                                       "border:2px solid white;border-radius:10px;\n"
                                       "}\n"
                                       "QPushButton:pressed{\n"
                                       " font-size:26px;  \n"
                                       "color:#f2f2f2;  \n"
                                       " font-family: 微软雅黑,宋体,Arial,Helvetica,Verdana,sans-serif;  \n"
                                       "background-color: rgba(95, 255, 37, 255);\n"
                                       "border:2px solid white;border-radius:10px;\n"
                                       "padding-left:3px;\n"
                                       "padding-top:3px;\n"
                                       "}"))
        runbtn.setCheckable(True)
        runbtn.setAutoExclusive(True)
        runbtn.clicked.connect(self.execution)

        vBox = QVBoxLayout()
        vBox.addWidget(uploadbtn)
        vBox.addWidget(clearbtn)
        vBox.addWidget(runbtn)

        hBox = QHBoxLayout()
        hBox.addWidget(self.present)
        hBox.addLayout(vBox)
        plate = QPalette()
        plate.setColor(QPalette.Background, QColor(185,211,244))

        self.setPalette(plate)

        self.setLayout(hBox)
        self.show()

    def clearinputimage(self):
        self.cnt = 0
        for root, dirs, files in os.walk('./images'):
            for name in files:
                os.remove(os.path.join(root, name))

    def uploadinputimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "(*.jpg *.png)")
        if imgName == '':
            return
        shutil.copy(imgName,'./images/'+str(self.cnt)+imgName[-4:])
        self.cnt += 1
        img = QPixmap(imgName)# .scaled(self.present.width(), self.present.height())
        self.present.setPixmap(img)


    def execution(self):
        os.system("python setup.py")


if __name__ == '__main__':

    app = QApplication(sys.argv)

    tmp = WINDOW()

    sys.exit(app.exec_())
