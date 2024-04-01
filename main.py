import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QErrorMessage,
    QInputDialog
)
from PyQt5.QtGui import (
    QIcon,
    QFont,
    QPixmap,
    QImage
)
from pathlib import Path
import cv2
from numpy import array as nparray
from matplotlib import pyplot as plt
from enum import Enum

class Kernels(Enum):
    Identity = [[0, 0, 0],[0, 1, 0],[0, 0, 0]]
    Edge = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    Sharpen = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    AverageBlur = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    GaussianBlur = [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]
    Custom = []

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('DIP Assignment 2 & 3')
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1200, 800)

        # image selection
        image_browse = QPushButton('Browse')
        image_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()
        self.filename_edit.setDisabled(True)
        image_select_hbox = QHBoxLayout()
        image_select_hbox.addWidget(QLabel('Selected Image:'))
        image_select_hbox.addWidget(self.filename_edit)
        image_select_hbox.addWidget(image_browse)
        self.restore_image_btn = QPushButton('Restore')
        self.restore_image_btn.clicked.connect(self.restore_image)
        self.restore_image_btn.setDisabled(True)
        image_select_hbox.addWidget(self.restore_image_btn)
        
        # image display 
        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # kernel
        self.kernel_dropdown = QComboBox()
        self.kernel_dropdown.addItems([k.name for k in Kernels])
        self.kernel_dropdown.currentTextChanged.connect(self.kernel_dropdown_changed)
        self.kernel_dropdown.destroyed.connect(self.kernel_dropdown_changed)
        kernel_select_vbox = QVBoxLayout()
        kernel_select_vbox.addWidget(QLabel('Select Kernel:'))
        ksel_hbox = QHBoxLayout()
        ksel_hbox.addWidget(self.kernel_dropdown)
        reset_kernel_btn = QPushButton()
        reset_kernel_btn.setIcon(QIcon(self.resource_path('undo-arrow.png')))
        reset_kernel_btn.clicked.connect(self.reset_kernel)
        reset_kernel_btn.setMaximumSize(30, 50)
        ksel_hbox.addWidget(reset_kernel_btn)        
        kernel_select_vbox.addLayout(ksel_hbox)
        self.kernel = nparray(Kernels.Identity.value)
        self.kernel_label = QLabel()
        self.kernel_label.setFont(QFont('Courier New', 11))
        self.kernel_label.setText(str(self.kernel))
        kernel_select_vbox.addWidget(self.kernel_label)
        self.persist_kernel = QCheckBox('Changes persist?', self)
        self.persist_kernel.setChecked(True)
        kernel_select_vbox.addWidget(self.persist_kernel)
        self.apply_kernel_btn = QPushButton('Apply')
        self.apply_kernel_btn.setDisabled(True)
        self.apply_kernel_btn.clicked.connect(self.apply_kernel)
        kernel_select_vbox.addWidget(self.apply_kernel_btn)
        kernel_group_box = QGroupBox('Filtering')
        kernel_group_box.setLayout(kernel_select_vbox)
        
        # thresholding
        self.thresh_dropdown = QComboBox()
        self.thresh_dropdown.addItems([
            "Global", 
            "Adaptive (Mean)", 
            "Adaptive (Gaussian)", 
            "Otsu", 
            "Otsu (with Gaussian Filter)"
        ])
        self.thresh_dropdown.currentIndexChanged.connect(self.thresh_dropdown_changed)
        thresh_vbox = QVBoxLayout()
        thresh_vbox.addWidget(self.thresh_dropdown)
        global_thresh_hbox = QHBoxLayout()
        global_thresh_hbox.addWidget(QLabel('Global Threshold:'))
        self.global_thresh_spin = QSpinBox()
        self.global_thresh_spin.setMinimum(0)
        self.global_thresh_spin.setMaximum(255)
        self.global_thresh_spin.setValue(127)
        global_thresh_hbox.addWidget(self.global_thresh_spin)
        thresh_vbox.addLayout(global_thresh_hbox)
        self.persist_thresh = QCheckBox('Changes persist?', self)
        self.persist_thresh.setChecked(False)
        thresh_vbox.addWidget(self.persist_thresh)
        self.apply_thresh_btn = QPushButton('Apply')
        self.apply_thresh_btn.setDisabled(True)
        self.apply_thresh_btn.clicked.connect(self.apply_thresh)
        thresh_vbox.addWidget(self.apply_thresh_btn)
        thresh_group_box = QGroupBox('Thresholding')
        thresh_group_box.setLayout(thresh_vbox)
        
        # histogram
        self.hist_label = QLabel()
        self.hist_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hist_label.setMinimumHeight(370)
        hist_vbox = QVBoxLayout()
        hist_vbox.addWidget(self.hist_label)
        self.save_hist_btn = QPushButton('Save Histogram')
        self.save_hist_btn.setDisabled(True)
        self.save_hist_btn.clicked.connect(self.save_hist_dialog)
        hist_vbox.addWidget(self.save_hist_btn)
        hist_group_box = QGroupBox('Histogram')
        hist_group_box.setLayout(hist_vbox)
        hist_group_box.setFixedWidth(400)
        
        main_layout = QGridLayout()
        main_layout.addLayout(image_select_hbox, 0, 0)
        main_layout.addWidget(self.image_label, 1, 0)
        op_vbox = QVBoxLayout()
        op_vbox.addWidget(kernel_group_box)
        op_vbox.addWidget(thresh_group_box)
        op_vbox.addWidget(hist_group_box)
        op_vbox.addStretch()
        self.save_img_btn = QPushButton('Save Image')
        self.save_img_btn.setDisabled(True)
        self.save_img_btn.clicked.connect(self.save_file_dialog)
        op_vbox.addWidget(self.save_img_btn)
        op_vbox.addWidget(QLabel('Made using PyQt5, OpenCV, and ♥\n[2024] Priyansh Agrahari'))
        main_layout.addLayout(op_vbox, 0, 1, 2, 1)
        self.setLayout(main_layout)
        self.show()

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select an Image for Processing", 
            "", 
            "Image Files (*.png *.jpg)"
        )
        if filename:
            self.image_path = Path(filename)
            self.filename_edit.setText(str(self.image_path))
            self.image = cv2.imread(str(self.image_path))
            self.isBGR = True
            self.set_frame_image(self.image)
            self.set_hist(self.image)
            self.restore_image_btn.setDisabled(False)
            self.save_img_btn.setDisabled(False)
            self.save_hist_btn.setDisabled(False)
            self.apply_kernel_btn.setDisabled(False)
            self.apply_thresh_btn.setDisabled(False)
    
    def save_file_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Save Current Image",
            "",
            "Image Files (*.png *.jpg)"
        )
        if filename:
            cv2.imwrite(filename, self.currentDispImage)
            msg = QMessageBox()
            msg.setWindowTitle('Image Saved')
            msg.setText('Image stored to disk successfully!')
            msg.setIcon(QMessageBox.Information)
            msg.exec()
    
    def save_hist_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Save Current Histogram",
            "",
            "Image Files (*.png *.jpg)"
        )
        if filename:
            cv2.imwrite(filename, self.currentHistImage)
            msg = QMessageBox()
            msg.setWindowTitle('Histogram Saved')
            msg.setText('Histogram stored to disk successfully!')
            msg.setIcon(QMessageBox.Information)
            msg.exec()
    
    def restore_image(self):
        if self.image_path:
            self.image = cv2.imread(str(self.image_path))
            self.isBGR = True
            self.set_frame_image(self.image)
            self.set_hist(self.image)
            msg = QMessageBox()
            msg.setWindowTitle('Image Restored')
            msg.setText('Image restored from disk successfully!')
            msg.setIcon(QMessageBox.Information)
            msg.exec()
    
    def set_frame_image(self, image, isBGR = True):
        self.currentDispImage = image
        if isBGR:
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        else:
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(900, 780, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
    
    def set_hist(self, image, isBGR = True):
        if isBGR:
            color = ('b', 'g', 'r')
        else:
            color = (0, )
        fig, ax = plt.subplots()
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i,col in enumerate(color):
            histr = cv2.calcHist([image],[i],None,[256],[0,256])
            ax.plot(histr, color = col if col != 0 else 'k')
            ax.set_xlim([0,256])
        fig.tight_layout()
        fig.canvas.draw()
        himg = nparray(fig.canvas.renderer.buffer_rgba())
        himg = cv2.cvtColor(himg, cv2.COLOR_RGBA2RGB)
        self.currentHistImage = himg
        qimage = QImage(himg.data, himg.shape[1], himg.shape[0], QImage.Format_RGB888)            
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(370, 370, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.hist_label.setPixmap(pixmap)
        self.hist_label.setMinimumHeight(0)
            
    
    def kernel_dropdown_changed(self, s : str):
        if s == "Custom":
            self.show_custom_kernel_dialog()
        else:
            self.kernel = nparray(Kernels[s].value)
        self.kernel_label.setText(str(self.kernel))
    
    def show_custom_kernel_dialog(self):
        s, ok = QInputDialog().getMultiLineText(self, "Enter Custom Kernel", 
                                                        "Python format matrix:", 
                                                        "[[0,0,0],\n[0,1,0],\n[0,0,0]]")
        if ok and len(s) > 0:
            try:
                m = nparray(eval(s))
                if m.shape[1] == m.shape[0]:
                    self.kernel = m
                else:
                    self.show_input_error_dialog('Non-square matrix entered :(\nPlease try again')
                    self.show_custom_kernel_dialog()
            except:
                self.show_input_error_dialog('Invalid matrix entered :/\nPlease try again')
                self.show_custom_kernel_dialog()
    
    def show_input_error_dialog(self, msg : str):
        emsg = QErrorMessage(self)
        emsg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        emsg.setWindowTitle('Invalid Data Entered!')
        emsg.showMessage(msg)
        emsg.exec()
    
    def apply_kernel(self):
        if self.image_path is not None:
            filtered = cv2.filter2D(src=self.image, ddepth=-1, kernel=self.kernel)
            if (self.persist_kernel.isChecked()):
                self.image = filtered
            self.set_frame_image(filtered, self.isBGR)
            self.set_hist(filtered, self.isBGR)
            msg = QMessageBox()
            msg.setWindowTitle('Kernel Applied')
            msg.setText('Kernel applied successfully!')
            msg.setIcon(QMessageBox.Information)
            msg.exec()
    
    def reset_kernel(self):
        self.kernel = nparray(Kernels.Identity.value)
        self.kernel_label.setText(str(self.kernel))
        self.kernel_dropdown.setCurrentIndex(0)
        
    def thresh_dropdown_changed(self, i : int):
        self.global_thresh_spin.setDisabled(i != 0)
    
    def apply_thresh(self):
        gray = self.image
        if gray is None:
            print('GRAY IS NONE')
            return
        if self.isBGR:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        result = None
        index = self.thresh_dropdown.currentIndex()
        if index == 0:
            # apply global thresh
            thresh = self.global_thresh_spin.value()
            ret, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        elif index == 1:
            # apply adaptive mean
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif index == 2:
            # apply adaptive gaussian
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif index == 3:
            # apply otsu
            ret, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif index == 4:
            # apply otsu w gaussian filter
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if result is not None:
            if self.persist_thresh.isChecked():
                self.image = result
                self.isBGR = False
            self.set_frame_image(result, False)
            self.set_hist(result, False)
            msg = QMessageBox()
            msg.setWindowTitle('Threshold Applied')
            msg.setText('%s Threshold applied successfully!' % self.thresh_dropdown.currentText())
            msg.setIcon(QMessageBox.Information)
            msg.exec()
        else:
            print('THRESH RESULT IS NONE')
    
    # fix for missing icon in build
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
