from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from core_en import YoloPredictor

from pathlib import Path
from utils.rtsp_win import Window
import traceback
import json
import sys
import cv2
import os

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # Signal to send execution signal from main window to YOLO instance
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # Basic interface setup
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Rounded corners and transparency
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flags: hide window border
        UIFuncitons.uiDefinitions(self)  # Custom interface definitions

        # Initial page setup
        self.task = ''
        self.PageIndex = 1
        self.content.setCurrentIndex(self.PageIndex)
        self.pushButton_detect.clicked.connect(self.button_detect)
        self.pushButton_pose.clicked.connect(self.button_pose)
        self.pushButton_classify.clicked.connect(self.button_classify)
        self.pushButton_segment.clicked.connect(self.button_segment)
        self.pushButton_track.clicked.connect(self.button_track)

        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)

        self.src_home_button.clicked.connect(self.return_home)
        #################################### Image or Video ####################################
        # Display module shadow
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # YOLO-v8 thread
        self.yolo_predict = YoloPredictor()  # Create YOLO instance
        self.select_model = self.model_box.currentText()  # Default model
        
        self.yolo_thread = QThread()  # Create YOLO thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))        
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))      
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))    
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x))) 
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        self.Qtimer_ModelBox = QTimer(self)  # Timer: monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # IOU text box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # IOU slider
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # Confidence text box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # Confidence slider
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # Speed text box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # Speed slider

        # Initialize hint window
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # Select folder source
        self.src_file_button.clicked.connect(self.open_src_file)  # Select local file
        # Single file
        self.src_img_button.clicked.connect(self.open_src_img)  # Select local file
        # Start test button
        self.run_button.clicked.connect(self.run_or_continue)  # Pause/Start
        self.stop_button.clicked.connect(self.stop)  # Stop

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # Save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        #################################### Image or Video ####################################

        #################################### Camera ####################################
        # Display cam module shadow
        UIFuncitons.shadow_style(self, self.Class_QF_cam, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF_cam, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF_cam, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF_cam, QColor(64, 186, 193))

        # YOLO-v8-cam thread
        self.yolo_predict_cam = YoloPredictor()  # Create YOLO instance
        self.select_model_cam = self.model_box_cam.currentText()  # Default model
        
        self.yolo_thread_cam = QThread()  # Create YOLO thread
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(lambda c: self.show_status(c))        
        self.yolo_predict_cam.yolo2main_fps.connect(lambda c: self.fps_label_cam.setText(c))      
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))    
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c))) 
        self.yolo_predict_cam.yolo2main_progress.connect(self.progress_bar_cam.setValue(0))
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

        self.Qtimer_ModelBox_cam = QTimer(self)  # Timer: monitor model file changes every 2 seconds
        self.Qtimer_ModelBox_cam.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox_cam.start(2000)

        # Cam model parameters
        self.model_box_cam.currentTextChanged.connect(self.cam_change_model)     
        self.iou_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_spinbox_cam'))  # IOU text box
        self.iou_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_slider_cam'))  # IOU slider
        self.conf_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_spinbox_cam'))  # Confidence text box
        self.conf_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_slider_cam'))  # Confidence slider
        self.speed_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_spinbox_cam'))  # Speed text box
        self.speed_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_slider_cam'))  # Speed slider

        # Initialize hint window
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')
        self.Model_name_cam.setText(self.select_model_cam)
        
        # Select detection source
        self.src_cam_button.clicked.connect(self.cam_button)  # Select camera
        
        # Start test button
        self.run_button_cam.clicked.connect(self.cam_run_or_continue)  # Pause/Start
        self.stop_button_cam.clicked.connect(self.cam_stop)  # Stop
        
        # Other function buttons
        self.save_res_button_cam.toggled.connect(self.cam_is_save_res)  # Save image option
        self.save_txt_button_cam.toggled.connect(self.cam_is_save_txt)  # Save label option
        #################################### Camera ####################################
        #################################### RTSP ####################################
        self.src_rtsp_button.clicked.connect(self.rtsp_button)
        #################################### RTSP ####################################
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # Left navigation button
        # Initialization
        self.load_config()

    def return_home(self):
        # 0: image/video page
        # 1: home page
        # 2: camera page
        if self.PageIndex != 1:
            self.PageIndex = 1
        self.content.setCurrentIndex(1)
        self.yolo_predict_cam.source = ''
        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)          
        # Terminate thread if running
        self.yolo_thread_cam.quit() 
        self.cam_stop()
        self.yolo_thread.quit()
        self.stop()
        self.show_status("Welcome to the YOLOv8 detection system, please select Mode")

    def button_classify(self):
        self.task = 'Classify'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True)) # Top right settings button

        # Load model directory
        self.pt_list = os.listdir('./models/classify/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/classify/' + x))   # Sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/classify/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/classify/%s" % self.select_model_cam

        # Load camera model directory
        self.pt_list_cam = os.listdir('./models/classify/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/classify/' + x))   # Sort by file size
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Classify")

    def button_detect(self):
        self.task = 'Detect'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/detect/%s" % self.select_model_cam
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True)) # Top right settings button

        # Load model directory
        self.pt_list = os.listdir('./models/detect/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/detect/' + x))   # Sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/detect/%s" % self.select_model_cam

        # Load camera model directory
        self.pt_list_cam = os.listdir('./models/detect/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/detect/' + x))   # Sort by file size
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Detect")

    def button_pose(self):
        self.task = 'Pose'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/pose/%s" % self.select_model_cam
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True)) # Top right settings button

        # Load model directory
        self.pt_list = os.listdir('./models/pose/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/pose/' + x))   # Sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/pose/%s" % self.select_model_cam

        # Load camera model directory
        self.pt_list_cam = os.listdir('./models/pose/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/pose/' + x))   # Sort by file size
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Pose")

    def button_segment(self):
        self.task = 'Segment'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/segment/%s" % self.select_model_cam
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True)) # Top right settings button

        # Load model directory
        self.pt_list = os.listdir('./models/segment/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/segment/' + x))   # Sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/segment/%s" % self.select_model_cam

        # Load camera model directory
        self.pt_list_cam = os.listdir('./models/segment/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/segment/' + x))   # Sort by file size
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Segment")

    def button_track(self):
        self.task = 'Track'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/track/%s" % self.select_model_cam
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True)) # Top right settings button
        
        # Load model directory
        self.pt_list = os.listdir('./models/track/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/track/' + x))   # Sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/track/%s" % self.select_model_cam

        # Load camera model directory
        self.pt_list_cam = os.listdir('./models/track/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/track/' + x))   # Sort by file size
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Track")


    ####################################image or video####################################
    # Open local file
    def open_src_file(self):
        if self.task == 'Classify':
            self.show_status("Current page: image or video detection page, Mode: Classify")
        if self.task == 'Detect':
            self.show_status("Current page: image or video detection page, Mode: Detect")
        if self.task == 'Pose':
            self.show_status("Current page: image or video detection page, Mode: Pose")
        if self.task == 'Segment':
            self.show_status("Current page: image or video detection page, Mode: Segment")
        if self.task == 'Track':
            self.show_status("Current page: image or video detection page, Mode: Track")      

        # Terminate cam thread to save resources
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
            self.cam_stop()

        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # Top right settings button

        # Set the configuration file path
        config_file = 'config/fold.json'
        
        # Read the configuration file content
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        
        # Get the last opened folder path
        open_fold = config['open_fold']
        
        # Use the current working directory if the last opened folder does not exist
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        
        if self.task == 'Track':
            name = QFileDialog.getExistingDirectory(self, 'Select your Folder', open_fold)
        else:
            name = QFileDialog.getExistingDirectory(self, 'Select your Folder', open_fold)
        
        # If the user selects a file
        if name:
            # Set the selected file path as the source for yolo_predict
            self.yolo_predict.source = name
            
            # Display the folder load status
            self.show_status('Load folder: {}'.format(os.path.dirname(name)))
            
            # Update the last opened folder path in the configuration file
            config['open_fold'] = os.path.dirname(name)
            
            # Write the updated configuration file back to the file
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            
            # Stop detection
            self.stop()

    # Open local file
    def open_src_img(self):
        if self.task == 'Classify':
            self.show_status("Current page: image or video detection page, Mode: Classify")
        if self.task == 'Detect':
            self.show_status("Current page: image or video detection page, Mode: Detect")
        if self.task == 'Pose':
            self.show_status("Current page: image or video detection page, Mode: Pose")
        if self.task == 'Segment':
            self.show_status("Current page: image or video detection page, Mode: Segment")
        if self.task == 'Track':
            self.show_status("Current page: image or video detection page, Mode: Track")      

        # Terminate cam thread to save resources
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
            self.cam_stop()

        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # Top right settings button

        # Set the configuration file path
        config_file = 'config/fold.json'
        
        # Read the configuration file content
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        
        # Get the last opened folder path
        open_fold = config['open_fold']
        
        # Use the current working directory if the last opened folder does not exist
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        
        # Allow the user to select a picture or video file through a file dialog
        if self.task == 'Track':
            name, _ = QFileDialog.getOpenFileName(self, 'Video', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv)")
        else:
            name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        
        # If the user selects a file
        if name:
            # Set the selected file path as the source for yolo_predict
            self.yolo_predict.source = name
            
            # Display the file load status
            self.show_status('Load file: {}'.format(os.path.basename(name)))
            
            # Update the last opened folder path in the configuration file
            config['open_fold'] = os.path.dirname(name)
            
            # Write the updated configuration file back to the file
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            
            # Stop detection
            self.stop()

    # Display original image and detection result in main window
    @staticmethod
    def show_image(img_src, label):
        try:
            # Get the height, width, and number of channels of the original image
            ih, iw, _ = img_src.shape
            # Get the width and height of the label
            w = label.geometry().width()
            h = label.geometry().height()
            
            # Maintain the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # Convert the image to RGB format
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            
            # Convert the image data to a QPixmap object
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            
            # Display the image on the label
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # Handle exceptions and print error messages
            print(repr(e))

    # Control start/pause of detection
    def run_or_continue(self):
        # Check if the source of YOLO prediction is empty
        if self.yolo_predict.source == '':
            self.show_status('Before starting detection, please select an image or video source...')
            self.run_button.setChecked(False)
        else:
            # Set the stop flag of YOLO prediction to False
            self.yolo_predict.stop_dtc = False
            
            # If the start button is checked
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # Activate button
                self.save_txt_button.setEnabled(False)  # Disable saving after detection starts
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')           
                self.yolo_predict.continue_dtc = True   # Control YOLO pause
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()
            # If the start button is not checked, it means pause detection
            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Detection paused...")
                self.run_button.setChecked(False)  # Stop button

    # Save test result button -- image/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            # Show message, indicating that image results will not be saved
            self.show_status('NOTE: Image results will not be saved')
            
            # Set the flag for saving results in the YOLO instance to False
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            # Show message, indicating that image results will be saved
            self.show_status('NOTE: Image results will be saved')
            
            # Set the flag for saving results in the YOLO instance to True
            self.yolo_predict.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            # Show message, indicating that label results will not be saved
            self.show_status('NOTE: Label results will not be saved')
            
            # Set the flag for saving labels in the YOLO instance to False
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            # Show message, indicating that label results will be saved
            self.show_status('NOTE: Label results will be saved')
            
            # Set the flag for saving labels in the YOLO instance to True
            self.yolo_predict.save_txt = True

    # Terminate button and related state handling
    def stop(self):
        # If the YOLO thread is running, terminate the thread
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit() # Terminate thread
        
        # Set the termination flag of the YOLO instance to True
        self.yolo_predict.stop_dtc = True
        
        # Restore the status of the start button
        self.run_button.setChecked(False)  
        
        self.save_res_button.setEnabled(True)   
        self.save_txt_button.setEnabled(True)

        # Clear the image in the prediction result display area
        self.pre_video.clear()           
        
        # Clear the image in the detection result display area
        self.res_video.clear()           
        
        # Set the value of the progress bar to 0
        self.progress_bar.setValue(0)
        
        # Reset the class number, target number, and fps labels
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            # If the value of the iou_spinbox changes, change the value of the iou_slider
            self.iou_slider.setValue(int(x * 100))

        elif flag == 'iou_slider':
            # If the value of the iou_slider changes, change the value of the iou_spinbox
            self.iou_spinbox.setValue(x / 100)
            # Show message, indicating IOU threshold change
            self.show_status('IOU Threshold: %s' % str(x / 100))
            # Set the IOU threshold of the YOLO instance
            self.yolo_predict.iou_thres = x / 100

        elif flag == 'conf_spinbox':
            # If the value of the conf_spinbox changes, change the value of the conf_slider
            self.conf_slider.setValue(int(x * 100))

        elif flag == 'conf_slider':
            # If the value of the conf_slider changes, change the value of the conf_spinbox
            self.conf_spinbox.setValue(x / 100)
            # Show message, indicating Confidence threshold change
            self.show_status('Conf Threshold: %s' % str(x / 100))
            # Set the Confidence threshold of the YOLO instance
            self.yolo_predict.conf_thres = x / 100

        elif flag == 'speed_spinbox':
            # If the value of the speed_spinbox changes, change the value of the speed_slider
            self.speed_slider.setValue(x)

        elif flag == 'speed_slider':
            # If the value of the speed_slider changes, change the value of the speed_spinbox
            self.speed_spinbox.setValue(x)
            # Show message, indicating delay time change
            self.show_status('Delay: %s ms' % str(x))
            # Set the delay time threshold of the YOLO instance
            self.yolo_predict.speed_thres = x  # milliseconds
    
    # Change model
    def change_model(self, x):
        # Get the currently selected model name
        self.select_model = self.model_box.currentText()
        
        # Set the new model name of the YOLO instance
        if self.task == 'Classify':
            self.yolo_predict.new_model_name = "./models/classify/%s" % self.select_model
        elif self.task == 'Detect':
            self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        elif self.task == 'Pose':
            self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        elif self.task == 'Segment':
            self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        elif self.task == 'Track':
            self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        # Show message, indicating model has been changed
        self.show_status('Change Model: %s' % self.select_model)
        
        # Display the new model name on the interface
        self.Model_name.setText(self.select_model)
    ####################################image or video####################################

    ####################################camera####################################
    def cam_button(self):
        self.yolo_predict_cam.source = 0
        self.show_status('Current page: Webcam detection page')
        # Terminate image or video thread to save resources
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            self.yolo_thread.quit()
            self.yolo_thread_cam.quit()
            self.stop()
            self.cam_stop()

        if self.PageIndex != 2:
            self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))  # Top right settings button

    # Control cam start/pause detection
    def cam_run_or_continue(self):
        if self.yolo_predict_cam.source == '':
            self.show_status('Camera not detected')
            self.run_button_cam.setChecked(False)

        else:
            # Set the stop flag for YOLO prediction to False
            self.yolo_predict_cam.stop_dtc = False

            # If the start button is checked
            if self.run_button_cam.isChecked():
                self.run_button_cam.setChecked(True)  # Start button
                self.save_txt_button_cam.setEnabled(False)  # Disable save after detection starts
                self.save_res_button_cam.setEnabled(False)
                self.show_status('Detection in progress...')
                self.yolo_predict_cam.continue_dtc = True  # Control whether YOLO is paused

                if not self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.start()
                    self.main2yolo_begin_sgl.emit()

            # If the start button is not checked, it means pause detection
            else:
                self.yolo_predict_cam.continue_dtc = False
                self.show_status("Detection paused...")
                self.run_button_cam.setChecked(False)  # Stop button

    # Display original image and detection results in the main cam window
    @staticmethod
    def cam_show_image(img_src, label, instance=None):
        try:
            # Get the height, width, and channels of the original image
            ih, iw, _ = img_src.shape

            # Get the width and height of the label
            w = label.geometry().width()
            h = label.geometry().height()

            # Maintain the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # Convert the image to RGB format
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)

            # Convert the image data to a QPixmap object
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)

            # Display the image on the label
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # Handle exceptions, print error messages
            traceback.print_exc()
            print(f"Error: {e}")
            if instance is not None:
                instance.show_status('%s' % e)

    # Change detection parameters
    def cam_change_val(self, c, flag):
        if flag == 'iou_spinbox_cam':
            # If the value of iou_spinbox changes, change the value of iou_slider
            self.iou_slider_cam.setValue(int(c * 100))

        elif flag == 'iou_slider_cam':
            # If the value of iou_slider changes, change the value of iou_spinbox
            self.iou_spinbox_cam.setValue(c / 100)
            # Display message indicating IOU threshold change
            self.show_status('IOU Threshold: %s' % str(c / 100))
            # Set the IOU threshold for the YOLO instance
            self.yolo_predict_cam.iou_thres = c / 100

        elif flag == 'conf_spinbox_cam':
            # If the value of conf_spinbox changes, change the value of conf_slider
            self.conf_slider_cam.setValue(int(c * 100))

        elif flag == 'conf_slider_cam':
            # If the value of conf_slider changes, change the value of conf_spinbox
            self.conf_spinbox_cam.setValue(c / 100)
            # Display message indicating confidence threshold change
            self.show_status('Conf Threshold: %s' % str(c / 100))
            # Set the confidence threshold for the YOLO instance
            self.yolo_predict_cam.conf_thres = c / 100

        elif flag == 'speed_spinbox_cam':
            # If the value of speed_spinbox changes, change the value of speed_slider
            self.speed_slider_cam.setValue(c)

        elif flag == 'speed_slider_cam':
            # If the value of speed_slider changes, change the value of speed_spinbox
            self.speed_spinbox_cam.setValue(c)
            # Display message indicating delay time change
            self.show_status('Delay: %s ms' % str(c))
            # Set the delay time threshold for the YOLO instance
            self.yolo_predict_cam.speed_thres = c  # milliseconds

    # Change model
    def cam_change_model(self, c):
        # Get the currently selected model name
        self.select_model_cam = self.model_box_cam.currentText()

        # Set the new model name for the YOLO instance
        if self.task == 'Classify':
            self.yolo_predict_cam.new_model_name = "./models/classify/%s" % self.select_model_cam
        elif self.task == 'Detect':
            self.yolo_predict_cam.new_model_name = "./models/detect/%s" % self.select_model_cam
        elif self.task == 'Pose':
            self.yolo_predict_cam.new_model_name = "./models/pose/%s" % self.select_model_cam
        elif self.task == 'Segment':
            self.yolo_predict_cam.new_model_name = "./models/segment/%s" % self.select_model_cam
        elif self.task == 'Track':
            self.yolo_predict_cam.new_model_name = "./models/track/%s" % self.select_model_cam
        # Display message indicating the model has changed
        self.show_status('Change Model：%s' % self.select_model_cam)

        # Display the new model name on the interface
        self.Model_name_cam.setText(self.select_model_cam)

    # Save test result button -- image/video
    def cam_is_save_res(self):
        if self.save_res_button_cam.checkState() == Qt.CheckState.Unchecked:
            # Display message, indicating that webcam results will not be saved
            self.show_status('NOTE: Webcam results will not be saved')

            # Set the flag for saving results in the YOLO instance to False
            self.yolo_thread_cam.save_res = False
        elif self.save_res_button_cam.checkState() == Qt.CheckState.Checked:
            # Display message, indicating that webcam results will be saved
            self.show_status('NOTE: Webcam results will be saved')

            # Set the flag for saving results in the YOLO instance to True
            self.yolo_thread_cam.save_res = True

    # Save test result button -- label (txt)
    def cam_is_save_txt(self):
        if self.save_txt_button_cam.checkState() == Qt.CheckState.Unchecked:
            # Display message, indicating that label results will not be saved
            self.show_status('NOTE: Label results will not be saved')

            # Set the flag for saving labels in the YOLO instance to False
            self.yolo_thread_cam.save_txt_cam = False
        elif self.save_txt_button_cam.checkState() == Qt.CheckState.Checked:
            # Display message, indicating that label results will be saved
            self.show_status('NOTE: Label results will be saved')

            # Set the flag for saving labels in the YOLO instance to True
            self.yolo_thread_cam.save_txt_cam = True

    # Stop cam button and related state processing
    def cam_stop(self):
        # If the YOLO thread is running, terminate the thread
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit() # Terminate thread

        # Set the termination flag in the YOLO instance to True
        self.yolo_predict_cam.stop_dtc = True

        # Restore the status of the run button
        self.run_button_cam.setChecked(False)

        # Enable save button permissions
        if self.task == 'Classify': 
            self.save_res_button_cam.setEnabled(False)
            self.save_txt_button_cam.setEnabled(False)
        else:
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)

        # Clear the predicted result display area image
        self.pre_cam.clear()

        # Clear the detection result display area image
        self.res_cam.clear()

        # Reset the progress bar value
        # self.progress_bar.setValue(0)

        # Reset the class number, target number, and fps labels
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')
    ####################################camera####################################
    ####################################rtsp####################################
    # RTSP input address
    def rtsp_button(self):
        # Terminate image or video thread to save resources
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            self.yolo_thread.quit() # Terminate thread
            self.yolo_thread_cam.quit()
            self.stop()
            self.cam_stop()

        self.content.setCurrentIndex(2)
        self.show_status('Current page: RTSP detection page')
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))   # Top right settings button

    # Load network source
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='NOTE', text='Loading RTSP...', time=1000, auto=True).exec()
            self.yolo_predict_cam.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loaded RTSP: {}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)
    ####################################rtsp####################################
    ####################################common####################################
    # Display information in the bottom status bar
    def show_status(self, msg):
        # Set the text of the status bar
        self.status_bar.setText(msg)
        if self.PageIndex == 0:
            # Perform corresponding operations based on different status messages
            if msg == 'Detection completed!':
                # Enable the save result and save text buttons
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                
                # Set the detection switch button to unchecked
                self.run_button.setChecked(False)    
                
                # Set the progress bar value to 0
                self.progress_bar.setValue(0)
                
                # If the YOLO thread is running, terminate it
                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()  # Terminate processing

            elif msg == 'Detection terminated!':
                # Enable the save result and save text buttons
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                
                # Set the detection switch button to unchecked
                self.run_button.setChecked(False)    
                
                # Set the progress bar value to 0
                self.progress_bar.setValue(0)
                
                # If the YOLO thread is running, terminate it
                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()  # Terminate processing
                    
                # Clear image display
                self.pre_video.clear()  # Clear original image
                self.res_video.clear()  # Clear detection result image
                self.Class_num.setText('--')  # Displayed class number
                self.Target_num.setText('--')  # Displayed target number
                self.fps_label.setText('--')  # Displayed frame rate information
                
        if self.PageIndex == 2:
            # Perform corresponding operations based on different status messages
            if msg == 'Detection completed!':
                # Enable the save result and save text buttons
                self.save_res_button_cam.setEnabled(True)
                self.save_txt_button_cam.setEnabled(True)
                
                # Set the detection switch button to unchecked
                self.run_button_cam.setChecked(False)    
                
                # Set the progress bar value to 0
                self.progress_bar_cam.setValue(0)
                
                # If the YOLO thread is running, terminate it
                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()  # Terminate processing

            elif msg == 'Detection terminated!':
                # Enable the save result and save text buttons
                self.save_res_button_cam.setEnabled(True)
                self.save_txt_button_cam.setEnabled(True)
                
                # Set the detection switch button to unchecked
                self.run_button_cam.setChecked(False)    
                
                # Set the progress bar value to 0
                self.progress_bar_cam.setValue(0)
                
                # If the YOLO thread is running, terminate it
                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()  # Terminate processing

                # Clear image display
                self.pre_cam.clear()  # Clear original image
                self.res_cam.clear()  # Clear detection result image
                self.Class_num_cam.setText('--')  # Displayed class number
                self.Target_num_cam.setText('--')  # Displayed target number
                self.fps_label_cam.setText('--')  # Displayed frame rate information

    # Monitor model file changes in a loop
    def ModelBoxRefre(self):
        # Get all model files in the model folder
        if self.task == 'Classify':
            pt_list = os.listdir('./models/classify')
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize('./models/classify/' + x))

            # If the model file list changes, update the model dropdown content
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        # Repeat the same process for other tasks ('Detect', 'Pose', 'Segment', 'Track')

    # Get the mouse position (for dragging the window by holding the title bar)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize adjustments when resizing the window (for resizing by dragging the bottom right corner)
    def resizeEvent(self, event):
        # Update the resize grip
        UIFuncitons.resize_grips(self)

    # Configuration initialization
    def load_config(self):
        config_file = 'config/setting.json'
        
        # If the configuration file does not exist, create it and write default settings
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            save_res_cam = 0
            save_txt_cam = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt,
                          "save_res": save_res_cam,
                          "save_txt": save_txt_cam
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            # If the configuration file exists, read the settings
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            
            # Check if the settings are complete, if not, use default values
            if len(config) != 7:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
                save_res_cam = 0
                save_txt_cam = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
                save_res_cam = config['save_res_cam']
                save_txt_cam = config['save_txt_cam']
        
        # Set the state of interface elements based on the settings
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)

        self.save_res_button_cam.setCheckState(Qt.CheckState(save_res_cam))
        self.yolo_predict_cam.save_res_cam = (False if save_res_cam == 0 else True)
        self.save_txt_button_cam.setCheckState(Qt.CheckState(save_txt_cam)) 
        self.yolo_predict_cam.save_txt_cam = (False if save_txt_cam == 0 else True)
        self.run_button_cam.setChecked(False)
        self.show_status("Welcome to use YOLOv8 detection system, please select Mode")

    # Close event, exit threads, save settings
    def closeEvent(self, event):
        # Save settings to the configuration file
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config['save_res_cam'] = (0 if self.save_res_button_cam.checkState()==Qt.Unchecked else 2)
        config['save_txt_cam'] = (0 if self.save_txt_button_cam.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        
        # Exit threads and the application
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            # If YOLO thread is running, terminate the thread
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()

            self.yolo_predict_cam.stop_dtc = True
            self.yolo_thread_cam.quit()            
            # Display exit message, wait for 3 seconds
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            
            # Exit the application
            sys.exit(0)
        else:
            # If YOLO thread is not running, exit the application directly
            sys.exit(0)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file = event.mimeData().urls()[0].toLocalFile()
        if file:
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.yolo_predict.source = Foldername
                self.show_image(self.yolo_predict.source[0], self.pre_video, 'path')
                self.show_status('Loaded Folder：{}'.format(os.path.basename(file)))
            else:
                self.yolo_predict.source = file
                if ".avi" or ".mp4" in self.yolo_predict.source:
                    self.cap = cv2.VideoCapture(self.yolo_predict.source)
                    ret, frame = self.cap.read()
                    if ret:
                        self.show_image(frame, self.pre_video, 'img')
                else:
                    self.show_image(self.yolo_predict.source, self.pre_video, 'path')
                self.show_status('Loaded File：{}'.format(os.path.basename(self.yolo_predict.source)))
    ####################################common####################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    # 創建相機線程
    # camera_thread = CameraThread()
    # camera_thread.imageCaptured.connect(Home.cam_data)
    # camera_thread.start()
    Home.show()
    sys.exit(app.exec())
