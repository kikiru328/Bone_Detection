# -----------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import os
import sys
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QTimer, QTime, QByteArray, Qt, QDateTime
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QWidget, QMainWindow, QLabel, QPushButton, QApplication, QFileDialog, QMessageBox, QDialog, QFrame
from PyQt5.uic import loadUi
import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime
import bone1 as bone

# weight path --------------------------------------------------------------------
# model_path = './weight/model.pt'
# tjnet_path = './weight/tjnet24.h5'

#  form --------------------------------------------------------------------------
form_secondwindow =uic.loadUiType("dashboard1.ui")[0]

#  dataframe ----------------------------------------------------------------
lms_df = pd.read_csv('./data/height_df.csv')
df_m = pd.read_csv('./data/male_year.csv',index_col='AGE')
df_fm = pd.read_csv('./data/female_year.csv',index_col='AGE')

# ui -------------------------------------------------------------------------

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./data/boneage_icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background-color: rgb(247, 242, 231);\n"
"border-color: rgb(121, 122, 126);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(25, 20, 950, 700))
        self.frame.setStyleSheet("color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(10)
        self.frame.setObjectName("frame")
        self.time = QtWidgets.QDateTimeEdit(self.frame)
        self.time.setGeometry(QtCore.QRect(10, 10, 194, 22))
        self.time.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.time.setFont(font)
        self.time.setStyleSheet("color: rgb(047, 079, 079);")
        self.time.setWrapping(False)
        self.time.setFrame(False)
        self.time.setAlignment(QtCore.Qt.AlignCenter)
        self.time.setReadOnly(True)
        self.time.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.time.setDateTime(QDateTime.currentDateTime())
        self.time.setObjectName("time")
        self.progressBar = QtWidgets.QProgressBar(self.frame)
        self.progressBar.setEnabled(False)
        self.progressBar.setGeometry(QtCore.QRect(570, 660, 290, 23))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        self.progressBar.setFont(font)
        self.progressBar.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.progressBar.setFocusPolicy(QtCore.Qt.NoFocus)
        self.progressBar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.progressBar.setToolTipDuration(-1)
        self.progressBar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar.setStyleSheet("color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(121, 122, 126));")
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setTextVisible(True)
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.title_1 = QtWidgets.QLabel(self.frame)
        self.title_2 = QtWidgets.QLabel(self.frame)
        self.title_1.setGeometry(QtCore.QRect(100, 150, 741, 80)) # 시작위치 x,y / 너비,높이
        self.title_2.setGeometry(QtCore.QRect(100, 230, 741, 100)) # 시작위치 x,y / 너비,높이
        font_1 = QtGui.QFont()
        font_2 = QtGui.QFont()
        font_1.setFamily("Calibri")
        font_2.setFamily("Calibri")
        font_1.setPointSize(52)
        font_2.setPointSize(60)
        font_1.setBold(True)
        font_2.setBold(True)
        font_1.setWeight(75)
        font_2.setWeight(75)
        self.title_1.setFont(font_1)
        self.title_2.setFont(font_2)
        self.title_1.setStyleSheet("color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(121, 122, 126));")
        self.title_2.setStyleSheet("color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(121, 122, 126));")
        self.title_1.setAlignment(QtCore.Qt.AlignCenter)
        self.title_2.setAlignment(QtCore.Qt.AlignCenter)
        self.title_1.setIndent(-1)
        self.title_2.setIndent(-1)
        self.title_1.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.title_2.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.title_1.setObjectName("title_1")
        self.title_2.setObjectName("title_2")
        self.next_button = QtWidgets.QPushButton(self.frame)
        self.next_button.setGeometry(QtCore.QRect(570, 580, 291, 61))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.next_button.setFont(font)
        self.next_button.setStyleSheet("background-color:rgb(216, 211, 205);\n"
"border:none;\n"
"border-bottom: 2px solid rgb(35, 35, 35);\n"
"color: rgb(50, 50, 50);\n"
"border-bottom-right-radius: 15px;\n"
"border-bottom-left-radius: 15px;\n"
"border-top-right-radius: 15px;\n"
"border-top-left-radius: 15px;\n"
"\n"
"QPushButton#pushButton:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(194, 194, 194);\n"
"}")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/dashboard2.ui"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.next_button.setIcon(icon1)
        self.next_button.setAutoDefault(True)
        self.next_button.setDefault(False)
        self.next_button.setObjectName("next_button")
        self.bone_img = QtWidgets.QLabel(self.frame)
        self.bone_img.setGeometry(QtCore.QRect(380, 300, 681, 321))
        self.bone_img.setStyleSheet("image: url(./data/boneage_icon.png);\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(255, 255, 255, 0));")
        self.bone_img.setText("")
        self.bone_img.setObjectName("bone_img")
        self.bone_img.raise_()
        self.title_1.raise_()
        self.title_2.raise_()
        self.time.raise_()
        self.progressBar.raise_()
        self.next_button.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.next_button.setText("Loading")
        if self.next_button.text() == "Loading":
            self.progressbar_change(MainWindow)

        # self.progressBar.valueChanged['int'].connect(self.progressbar_change)
        # self.next_button.clicked.connect(self.progressbar_change)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bone Age"))
        self.time.setDisplayFormat(_translate("MainWindow", "yyyy-MM-dd  hh:mm"))
        self.title_1.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" color:#797a7e;\">Bone Age</span></p></body></html>"))
        self.title_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" color:#797a7e;\">predictor</span></p></body></html>"))
        self.next_button.setText(_translate("MainWindow", "Loading"))

    def progressbar_change(self, MainWindow):

        while self.next_button.text() == "Loading":
            self.progressBar.setValue(np.random.randint(40,70))
            try :
                self.next_button.setText("Wait a minute")
                if self.next_button.text() == "Wait a minute":
                    self.progressBar.setValue(100)


                    print('True')

                else : print('progress wrong')

            except : 
                print('ERROR > check path.')
                break
    
#-----------------------------------------------------------------------------------------------------
class secondwindow(QDialog, QFrame, form_secondwindow):
    def __init__(self):
        super(secondwindow, self).__init__()
        loadUi("dashboard1.ui", self)
        self.setWindowIcon(QtGui.QIcon('./data/boneage_icon.png'))
        
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.time2.setFont(font)
        self.time2.setStyleSheet("color: rgb(047, 079, 079);")
        self.time2.setDateTime(QDateTime.currentDateTime())
        self.image_frame = QLabel(self)
        

    def filedialog_open(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open File', '',
                                            'All File(*);; Image File(*.png *.jpg)')
        
        if fname[0]:
            # QPixmap 객체
            global openpath
            openpath = fname[0]
            self.pixmap = QtGui.QPixmap(openpath)            
            self.pixmap = self.pixmap.scaled(481,621) # 이미지 스케일 변화
            self.image_frame.move(50,50) # 시작위치
            self.image_frame.setPixmap(self.pixmap)  # 이미지 세팅
            self.image_frame.setContentsMargins(0,0,0,0)
            self.image_frame.resize(481,621)  # 프레임 스케일

        else:
            QMessageBox.about(self, 'Warning', 'No file selected.')
              


    # info text 입력
    def info(self):
        global name_txt, age_txt, height_txt
        name_txt= self.input_name.text()
        age_txt= self.input_age.text()
        height_txt= self.input_height.text()
    
    # ok버튼 클릭시 third page 로 텍스트 넘기기 
    def push_ok_button(self):
        self.ok_button.setText("OK")
        try : 
            th.input_name2.setText(name_txt) 
            th.input_name2.setReadOnly(True)

        except : 
            QMessageBox.about(self, 'Warning', 'Enter your name.')
            
        try : 
            th.input_age2.setText(age_txt)
            th.input_age2.setReadOnly(True)

        except : 
            QMessageBox.about(self, 'Warning', 'Enter your Age.')
        
    def reset_info(self):
        se.input_name.setText("")
        se.input_age.setText("")
        se.input_height.setText("")

        th.input_name2.setText("") 
        th.input_age2.setText("")
        th.input_gender2.setText("")
        
        se.ok_button.setText("Insert")


    # Female, Male 버튼
    def female(self):
        global gender, gender_text
        gender = 0
        gender_text = 'Female'
    
    def male(self):
        global gender, gender_text
        gender = 1
        gender_text = 'Male'


    # # 다음페이지로 
    def gotonextpage(self):
        if se.ok_button.text() != "Insert":
            try:
                self.next_button2.setEnabled(False)
                self.progressBar2.setValue(np.random.randint(0,5))
                self.next_button2.setText("Loading")

                global filename, graph_path, formattedDate, pngname, now
                now = datetime.now()
                
                formattedDate = now.strftime("%Y%m%d_%H%M%S")
                filename = formattedDate +'.jpg'

                save_path = './img_save/' + filename
                graph_path = './graph_save/' + filename

                # global openpath
                openimg = bone.read_img(openpath)
                self.progressBar2.setValue(np.random.randint(10,20))
                mask = bone.make_mask(openimg)
                masked = bone.cut_mask(openimg, mask)
                self.progressBar2.setValue(np.random.randint(25,40))
                rotated_img = bone.img_rotation(masked)
                self.progressBar2.setValue(np.random.randint(50,60))
                bone_img = bone.Decomposing(rotated_img,60,55,50,25)
                self.progressBar2.setValue(np.random.randint(70,80))
                cv2.imwrite(save_path, bone_img)
            
                # global yolo
                crops, yoloimg, result = bone.yolo_crop_img(save_path, yolo)
                self.progressBar2.setValue(np.random.randint(95,100))
                h,w,c = yoloimg.shape
                widget.setCurrentIndex(widget.currentIndex()+1)
                
                qImg = QtGui.QImage(yoloimg, w, h, w*c, QtGui.QImage.Format_RGB888)
                th.yolo_img = QtGui.QPixmap.fromImage(qImg)
                th.yolo_img = th.yolo_img.scaled(411,521)
                th.yolo_frame.move(40,80)
                th.yolo_frame.setPixmap(th.yolo_img)  # 이미지 세팅
                th.yolo_frame.setContentsMargins(0,0,0,0)
                th.yolo_frame.resize(411,521)  # 프레임 스케일 

                try : 
                    X = bone.out_crop_img(crops, gender)
                    global prediction_BA
                    prediction_BA = bone.predict_zscore(X, tjnet)  
                    prediction_BA = prediction_BA.round(2)
                    # ---------------------------------------------------------------
                    th.input_gender2.setText(gender_text) 
                    th.input_gender2.setReadOnly(True)
                    th.pred.setText(f'{prediction_BA}') 
                    th.pred.setReadOnly(True)
                except : 
                    th.pred.setReadOnly(True)
                    th.pred.setText("Please enter your gender.")   
                
                global result_th, Predict_Height, df_m, df_fm, lms_df
                current_Height = float(height_txt)
                try : 
                    result_th, Predict_Height = bone.Height_graph(gender, prediction_BA, current_Height, df_m, df_fm, lms_df, graph_path)
                    th.result_th.setText(f'{Predict_Height}, ({result_th})')
                except Exception as e:
                    print(e)


            except: 
                QMessageBox.about(self, 'Warning', 'Enter information.')  
                self.next_button2.setText("NEXT >>")
                self.progressBar2.setValue(0)
                self.next_button2.setEnabled(True) 
            
        else : 
            QMessageBox.about(self, 'Warning', 'Enter information.')

#-----------------------------------------------------------------------------------------------------
class thirdwindow(QDialog):
    def __init__(self):
        super(thirdwindow, self).__init__()
        loadUi("dashboard2.ui", self)

        self.setWindowIcon(QtGui.QIcon('./data/boneage_icon.png'))
        font = QtGui.QFont()

        font.setFamily("Calibri")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.time3.setFont(font)
        self.time3.setStyleSheet("color: rgb(047, 079, 079);")
        self.time3.setDateTime(QDateTime.currentDateTime())
        self.before_button.clicked.connect(self.gotobeforepage)
        
        self.yolo_frame = QLabel(self)
    

    def gotobeforepage(self):
        widget.setCurrentIndex(widget.currentIndex()-1)
        se.next_button2.setText("NEXT >>")
        se.progressBar2.setValue(0)
        se.next_button2.setEnabled(True)
        se.ok_button.setText("Insert")
        
        th.print_button.setText("print>>")
        th.print_button.setEnabled(True)
        th.progressBar3.setValue(0)



    def show_excel(self):
        self.print_button.setEnabled(False)
        self.print_button.setText("Loading")
        self.progressBar3.setValue(np.random.randint(20,50))
        print('show_excel')

        global result_th, graph_path

        bone.print_excel_file(name_txt , gender_text , float(age_txt) , float(height_txt) , prediction_BA , result_th, Predict_Height, openpath, graph_path, now)
        print('save_done')
        self.progressBar3.setValue(100)
        self.print_button.setText("Save!")

    def show_exit(self):
        sys.exit(app.exec_())


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    widget = QtWidgets.QStackedWidget()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    # app.exec_()
    # 화면전환용 위젯 생성
    widget = QtWidgets.QStackedWidget()

    # 레이아웃 인스턴스 생성
    se = secondwindow()
    th = thirdwindow()
    
    # 위젯 추가
    widget.addWidget(se)
    widget.addWidget(th)
    
    # 프로그램 화면
    widget.setFixedHeight(800)
    widget.setFixedWidth(1000)
    
    widget.setWindowTitle('Bone Age')
    widget.setWindowIcon(QtGui.QIcon('./data/boneage_icon.jpg'))
    widget.setFixedWidth(1000)
    

    MainWindow.show()
    import torch
    import tensorflow.keras as tf
    model_path = './weight/model.pt'
    tjnet_path = './weight/tjnet24.h5'

    global tjnet
    tjnet = tf.models.load_model(tjnet_path, compile=False)

    global yolo
    yolo = torch.load(model_path, map_location='cpu')


    MainWindow.close()


    
    widget.show()
    

    sys.exit(app.exec_())
    # MainWindow.close()