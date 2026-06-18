import os
import sys
import json
import traceback
import cv2
import numpy as np
from pathlib import Path

# PySide6 component imports
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, Qt

# Project custom module imports
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from core import YoloPredictor
from utils.rtsp_win import Window


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # Main window sends execution signal to YOLO instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # 1. Base UI and frameless window configuration with rounded corners
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)
        UIFunctions.uiDefinitions(self)

        # 2. State and page index initialization
        self.task = ''
        self.PageIndex = 1
        self.dragPos = None
        self.content.setCurrentIndex(self.PageIndex)
        self.video_slider.setEnabled(False)

        # 3. Core inference thread and controller initialization
        self.yolo_predict = YoloPredictor()
        self.yolo_thread = QThread()
        self.yolo_predict_cam = YoloPredictor()
        self.yolo_thread_cam = QThread()

        # 4. Bind UI events and signals
        self._init_ui_connections()
        self._init_yolo_signals()
        self._init_yolo_cam_signals()

        # 5. Start the model folder monitoring timer (Merged into a single timer to save resources)
        self.pt_list = []
        self.pt_list_cam = []
        self.Qtimer_ModelBox = QTimer(self)
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # 6. Configuration initialization
        self.select_model = self.model_box.currentText()
        self.select_model_cam = self.model_box_cam.currentText()
        self.load_config()
        self.show_status("Welcome to YOLOv8 Detection System, please select a Mode")

    def _init_ui_connections(self):
        """Consolidates event bindings for all UI components."""
        # Mode switching buttons
        task_buttons = {
            self.pushButton_detect: ('Detect', ':/all/img/detect.png', ':/all/img/detect_hover.png'),
            self.pushButton_pose: ('Pose', ':/all/img/pose.png', ':/all/img/pose_hover.png'),
            self.pushButton_classify: ('Classify', ':/all/img/classify.png', ':/all/img/classify_hover.png'),
            self.pushButton_segment: ('Segment', ':/all/img/segment.png', ':/all/img/segment_hover.png'),
            self.pushButton_track: ('Track', ':/all/img/track.png', ':/all/img/track_hover.png')
        }
        for btn, (task_name, normal_icon, hover_icon) in task_buttons.items():
            btn.clicked.connect(lambda checked=False, t=task_name: self.switch_mode(t))
            UIFunctions.setup_button(btn, normal_icon, hover_icon)

        # Enhance layout panel drop-shadows
        shadows = [
            (self.Class_QF, QColor(162, 129, 247)), (self.Target_QF, QColor(251, 157, 139)),
            (self.Fps_QF, QColor(170, 128, 213)), (self.Model_QF, QColor(64, 186, 193)),
            (self.Class_QF_cam, QColor(162, 129, 247)), (self.Target_QF_cam, QColor(251, 157, 139)),
            (self.Fps_QF_cam, QColor(170, 128, 213)), (self.Model_QF_cam, QColor(64, 186, 193))
        ]
        for qf, color in shadows:
            UIFunctions.shadow_style(self, qf, color)

        # Basic navigation and stream control inputs
        self.src_home_button.clicked.connect(self.return_home)
        self.src_file_button.clicked.connect(self.open_src_file)
        self.src_img_button.clicked.connect(self.open_src_img)
        self.src_cam_button.clicked.connect(self.cam_button)
        self.src_rtsp_button.clicked.connect(self.rtsp_button)
        self.ToggleBotton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # Pipeline execution controls
        self.run_button.clicked.connect(self.run_or_continue)
        self.stop_button.clicked.connect(self.stop)
        self.run_button_cam.clicked.connect(self.cam_run_or_continue)
        self.stop_button_cam.clicked.connect(self.cam_stop)

        # File export output check boxes
        self.save_res_button.toggled.connect(self.is_save_res)
        self.save_txt_button.toggled.connect(self.is_save_txt)
        self.save_res_button_cam.toggled.connect(self.cam_is_save_res)
        self.save_txt_button_cam.toggled.connect(self.cam_is_save_txt)

        # Model and threshold dropdown connections
        self.model_box.currentTextChanged.connect(self.change_model)
        self.model_box_cam.currentTextChanged.connect(self.cam_change_model)

        # Synced hyperparameter adjustments using the consolidated generic lambda calls
        self.iou_spinbox.valueChanged.connect(lambda val: self.generic_change_val(val, 'iou', is_cam=False, src_type='spinbox'))
        self.iou_slider.valueChanged.connect(lambda val: self.generic_change_val(val, 'iou', is_cam=False, src_type='slider'))
        self.conf_spinbox.valueChanged.connect(lambda val: self.generic_change_val(val, 'conf', is_cam=False, src_type='spinbox'))
        self.conf_slider.valueChanged.connect(lambda val: self.generic_change_val(val, 'conf', is_cam=False, src_type='slider'))
        self.speed_spinbox.valueChanged.connect(lambda val: self.generic_change_val(val, 'speed', is_cam=False, src_type='spinbox'))
        self.speed_slider.valueChanged.connect(lambda val: self.generic_change_val(val, 'speed', is_cam=False, src_type='slider'))

        self.iou_spinbox_cam.valueChanged.connect(lambda val: self.generic_change_val(val, 'iou', is_cam=True, src_type='spinbox'))
        self.iou_slider_cam.valueChanged.connect(lambda val: self.generic_change_val(val, 'iou', is_cam=True, src_type='slider'))
        self.conf_spinbox_cam.valueChanged.connect(lambda val: self.generic_change_val(val, 'conf', is_cam=True, src_type='spinbox'))
        self.conf_slider_cam.valueChanged.connect(lambda val: self.generic_change_val(val, 'conf', is_cam=True, src_type='slider'))
        self.speed_spinbox_cam.valueChanged.connect(lambda val: self.generic_change_val(val, 'speed', is_cam=True, src_type='spinbox'))
        self.speed_slider_cam.valueChanged.connect(lambda val: self.generic_change_val(val, 'speed', is_cam=True, src_type='slider'))

        # Set default empty displays for text values
        for label in [self.Class_num, self.Target_num, self.fps_label, self.Class_num_cam, self.Target_num_cam, self.fps_label_cam]:
            label.setText('--')
        self.time_label_cam.setText('Program Runtime: 00:00:00')

    def _init_yolo_signals(self):
        """Initializes signal-slot connections for standard file/video pipeline inference."""
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video, 'img'))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video, 'img'))
        self.yolo_predict.yolo2main_status_msg.connect(self.show_status)
        self.yolo_predict.yolo2main_fps.connect(self.fps_label.setText)
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_time.connect(self.time_label.setText)
        self.yolo_predict.yolo2main_silder_range.connect(self.position_changed)
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

    def _init_yolo_cam_signals(self):
        """Initializes signal-slot connections for integrated webcam and network stream inference."""
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(self.show_status)
        self.yolo_predict_cam.yolo2main_fps.connect(self.fps_label_cam.setText)
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c)))
        self.yolo_predict_cam.yolo2main_time.connect(self.time_label_cam.setText)
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

    def generic_change_val(self, target_val, param_type, is_cam=False, src_type='slider'):
        """
        [Refactoring Optimization] Generic hyperparameter sync manager. Compresses 12 repetitive conditional branches
        into a single safe function to prevent maintenance dead-locks.
        """
        suffix = '_cam' if is_cam else ''
        predictor = self.yolo_predict_cam if is_cam else self.yolo_predict

        if param_type in ['iou', 'conf']:
            if src_type == 'slider':
                actual_val = target_val / 100
                getattr(self, f"{param_type}_spinbox{suffix}").setValue(actual_val)
            else:
                actual_val = target_val
                getattr(self, f"{param_type}_slider{suffix}").setValue(int(target_val * 100))
            
            setattr(predictor, f"{param_type}_thres", actual_val)
            self.show_status(f"{param_type.upper()} Threshold: {actual_val}")
        
        elif param_type == 'speed':
            if src_type == 'slider':
                getattr(self, f"speed_spinbox{suffix}").setValue(target_val)
            else:
                getattr(self, f"speed_slider{suffix}").setValue(target_val)
            predictor.speed_thres = target_val
            self.show_status(f"Delay: {target_val} ms")

    def switch_mode(self, task):
        self.task = task
        self.yolo_predict.task = task
        self.yolo_predict_cam.task = task

        self.update_model_lists()
        
        self.PageIndex = 0
        self.content.setCurrentIndex(0)
        
        # Enable all input control options in the sidebar
        for btn in [self.src_home_button, self.src_file_button, self.src_img_button, self.src_cam_button, self.src_rtsp_button, self.settings_button]:
            btn.setEnabled(True)
            
        self.time_label.setText('00:00:00 / 00:00:00')
        
        # Defensive disconnection to eliminate libpyside runtime connection warnings
        try:
            if self.settings_button.receivers(self.settings_button.clicked) > 0:
                self.settings_button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.settings_button.clicked.connect(lambda: UIFunctions.settingBox(self, True))
        self.show_status(f"Current Panel: Image or Video Window | Mode: {task}")

    def update_model_lists(self):
        model_dir = f'./models/{self.task.lower()}/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        pt_files = [file for file in os.listdir(model_dir) if file.endswith(('.pt', '.onnx', '.engine'))]
        pt_files.sort(key=lambda x: os.path.getsize(os.path.join(model_dir, x)))
        
        self.pt_list = pt_files
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        if self.select_model in self.pt_list:
            self.model_box.setCurrentText(self.select_model)
        self.yolo_predict.new_model_name = os.path.join(model_dir, self.model_box.currentText())

        self.pt_list_cam = pt_files.copy()
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        if self.select_model_cam in self.pt_list_cam:
            self.model_box_cam.setCurrentText(self.select_model_cam)
        self.yolo_predict_cam.new_model_name = os.path.join(model_dir, self.model_box_cam.currentText())

    def reset_yolo_thread(self):
        """[Thread Protection] Safely drops previous signal bindings before reallocating components to avoid memory leak."""
        if self.yolo_thread.isRunning():
            self.yolo_thread.requestInterruption()
            self.yolo_thread.quit()
            self.yolo_thread.wait()

        try:
            self.yolo_predict.disconnect()
        except RuntimeError:
            pass
            
        self.yolo_predict.deleteLater()
        self.yolo_predict = YoloPredictor()
        self.yolo_thread = QThread()
        self._init_yolo_signals()

    def reset_yolo_thread_cam(self):
        """[Thread Protection] Safely drops and reallocates the camera inference worker runtime thread."""
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.requestInterruption()
            self.yolo_thread_cam.quit()
            self.yolo_thread_cam.wait()

        try:
            self.yolo_predict_cam.disconnect()
        except RuntimeError:
            pass

        self.yolo_predict_cam.deleteLater()
        self.yolo_predict_cam = YoloPredictor()
        self.yolo_thread_cam = QThread()
        self._init_yolo_cam_signals()

    def reset(self):
        self.stop()
        self.reset_yolo_thread()
        self.cam_stop()
        self.reset_yolo_thread_cam()

    def return_home(self):
        for btn in [self.src_home_button, self.src_file_button, self.src_img_button, self.src_cam_button, self.src_rtsp_button, self.settings_button]:
            btn.setEnabled(False)

        self.PageIndex = 1
        self.yolo_predict.source = ''
        self.yolo_predict_cam.source = ''
        self.content.setCurrentIndex(1)
        self.reset()
        self.show_status("Welcome to YOLOv8 Detection System, please select a Mode")

    def open_src_file(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        
        self.reset()
        self.switch_mode(self.task)
            
        config_file = 'config/fold.json'
        os.makedirs('config', exist_ok=True)
        
        try:
            config = json.load(open(config_file, 'r', encoding='utf-8')) if os.path.exists(config_file) else {}
        except Exception:
            config = {}
            
        open_fold = config.get('open_fold', os.getcwd())
        FolderPath = QFileDialog.getExistingDirectory(self, 'Select Target Directory', open_fold)
        
        if FolderPath:
            file_formats = {".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2", ".mp4", ".avi"}
            all_files = os.listdir(FolderPath)
            folder_name = [os.path.join(FolderPath, f) for f in all_files if os.path.splitext(f)[1].lower() in file_formats]
            
            if folder_name:
                self.yolo_predict.source = folder_name
                self.show_status(f'Loaded Folder: {os.path.basename(FolderPath)}')
                config['open_fold'] = os.path.dirname(FolderPath)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                self.stop()
            else:
                self.show_status('No supported image or video files found inside the selected directory...')

    def open_src_img(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)

        self.reset()
        self.switch_mode(self.task)
            
        config_file = 'config/fold.json'
        try:
            config = json.load(open(config_file, 'r', encoding='utf-8')) if os.path.exists(config_file) else {}
        except Exception:
            config = {}
            
        open_fold = config.get('open_fold', os.getcwd())
        
        title = 'Select Video File' if self.task == 'Track' else 'Select Video or Image File'
        filters = "Video File(*.mp4 *.mkv *.avi *.flv)" if self.task == 'Track' else "Media File(*.mp4 *.mkv *.avi *.flv *.jpg *.jpeg *.png)"
        
        name, _ = QFileDialog.getOpenFileName(self, title, open_fold, filters)
        
        if name:
            self.yolo_predict.source = name
            self.show_status(f'Loaded File: {os.path.basename(name)}')
            config['open_fold'] = os.path.dirname(name)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            self.stop()

    @staticmethod
    def show_image(img_src, label, flag):
        if flag == "path":
            img_src = cv2.imdecode(np.fromfile(img_src, dtype=np.uint8), -1)

        if img_src is None or img_src.size == 0:
            return

        ih, iw, _ = img_src.shape
        w, h = label.geometry().width(), label.geometry().height()

        if iw / w > ih / h:
            scal = w / iw
            nw, nh = w, int(scal * ih)
        else:
            scal = h / ih
            nw, nh = int(scal * iw), h

        frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_or_continue(self):
        if not self.yolo_predict.source:
            self.show_status('Please select a valid image or video input stream before launching inference...')
            self.run_button.setChecked(False)
            return

        self.yolo_predict.stop_dtc = False

        if self.run_button.isChecked():
            self.save_txt_button.setEnabled(False)
            self.save_res_button.setEnabled(False)
            self.show_status('Detecting...')
            self.yolo_predict.continue_dtc = True

            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()
        else:
            self.yolo_predict.continue_dtc = False
            self.show_status("Detection Paused...")
            self.run_button.setChecked(False)

    def is_save_res(self):
        is_checked = self.save_res_button.checkState() == Qt.CheckState.Checked
        self.show_status('NOTE: Inference visual results will be exported and saved' if is_checked else 'NOTE: Inference visual results will NOT be saved')
        self.yolo_predict.save_res = is_checked

    def is_save_txt(self):
        is_checked = self.save_txt_button.checkState() == Qt.CheckState.Checked
        self.show_status('NOTE: Label text results will be exported and saved' if is_checked else 'NOTE: Label text results will NOT be saved')
        self.yolo_predict.save_txt = is_checked

    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()
        self.yolo_predict.stop_dtc = True

        self.run_button.setChecked(False)
        self.save_res_button.setEnabled(True)
        self.save_txt_button.setEnabled(True)
        self.pre_video.clear()
        self.res_video.clear()
        self.time_label.setText('00:00:00 / 00:00:00')
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        if not self.select_model:
            return
        model_prefix = f'./models/{self.task.lower()}/'
        self.yolo_predict.new_model_name = os.path.join(model_prefix, self.select_model)
        self.show_status(f'Change Model: {self.select_model}')
        self.Model_name.setText(self.select_model)

    def position_changed(self, progress):
        try:
            progress_list = progress.split(',')
            self.video_slider.setRange(0, int(progress_list[1]))
            self.video_slider.setValue(int(progress_list[0]))
        except Exception:
            pass

    def cam_button(self):
        self.reset()
        self.switch_mode(self.task)
                
        self.yolo_predict_cam.source = 0
        self.show_status('Current Panel: Webcam Execution Window')

        if self.PageIndex != 2:
            self.PageIndex = 2
        self.content.setCurrentIndex(2)
        
        try:
            if self.settings_button.receivers(self.settings_button.clicked) > 0:
                self.settings_button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.settings_button.clicked.connect(lambda: UIFunctions.cam_settingBox(self, True))

    def cam_run_or_continue(self):
        if self.yolo_predict_cam.source == '':
            self.show_status('No supported device/camera resource index found')
            self.run_button_cam.setChecked(False)
            return

        self.yolo_predict_cam.stop_dtc = False

        if self.run_button_cam.isChecked():
            self.run_button_cam.setChecked(True)
            self.save_txt_button_cam.setEnabled(False)
            self.save_res_button_cam.setEnabled(False)
            self.show_status('Detecting...')
            self.yolo_predict_cam.continue_dtc = True

            if not self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.start()
                self.main2yolo_begin_sgl.emit()
        else:
            self.yolo_predict_cam.continue_dtc = False
            self.show_status("Detection Paused...")
            self.run_button_cam.setChecked(False)

    @staticmethod
    def cam_show_image(img_src, label):
        if img_src is None or img_src.size == 0:
            return
        ih, iw, _ = img_src.shape
        w, h = label.geometry().width(), label.geometry().height()

        if iw / w > ih / h:
            scal = w / iw
            nw, nh = w, int(scal * ih)
        else:
            scal = h / ih
            nw, nh = int(scal * iw), h

        frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def cam_change_model(self, c):
        self.select_model_cam = self.model_box_cam.currentText()
        if not self.select_model_cam:
            return
        model_prefix = f'./models/{self.task.lower()}/'
        self.yolo_predict_cam.new_model_name = os.path.join(model_prefix, self.select_model_cam)
        self.show_status(f'Change Model: {self.select_model_cam}')
        self.Model_name_cam.setText(self.select_model_cam)

    def cam_is_save_res(self):
        is_checked = self.save_res_button_cam.checkState() == Qt.CheckState.Checked
        self.show_status('NOTE: Webcam results will be exported and saved' if is_checked else 'NOTE: Webcam results will NOT be saved')
        self.yolo_predict_cam.save_res_cam = is_checked

    def cam_is_save_txt(self):
        is_checked = self.save_txt_button_cam.checkState() == Qt.CheckState.Checked
        self.show_status('NOTE: Label results will be exported and saved' if is_checked else 'NOTE: Label results will NOT be saved')
        self.yolo_predict_cam.save_txt_cam = is_checked

    def cam_stop(self):
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
        self.yolo_predict_cam.stop_dtc = True

        self.run_button_cam.setChecked(False)
        self.save_res_button_cam.setEnabled(True)
        self.save_txt_button_cam.setEnabled(True)
        self.pre_cam.clear()
        self.res_cam.clear()
        self.time_label_cam.setText('Program Runtime: 00:00:00')
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')

    def rtsp_button(self):
        self.reset()
        self.switch_mode(self.task)
        
        self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.show_status('Current Panel: RTSP Stream Connection Panel')
                
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        os.makedirs('config', exist_ok=True)

        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({"ip": ip}, f, ensure_ascii=False, indent=2)
        else:
            try:
                config = json.load(open(config_file, 'r', encoding='utf-8'))
                ip = config.get('ip', "rtsp://admin:admin888@192.168.1.2:555")
            except Exception:
                ip = "rtsp://admin:admin888@192.168.1.2:555"

        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

        self.yolo_predict_cam.stream_buffer = True
        try:
            if self.settings_button.receivers(self.settings_button.clicked) > 0:
                self.settings_button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.settings_button.clicked.connect(lambda: UIFunctions.cam_settingBox(self, True))

    def load_rtsp(self, ip):
        try:
            MessageBox(self.close_button, title='Notice', text='Loading rtsp stream network connection...', time=1000, auto=True).exec()
            self.yolo_predict_cam.source = ip
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                json.dump({"ip": ip}, f, ensure_ascii=False, indent=2)
            self.show_status(f'Loading rtsp stream context URL: {ip}')
            self.rtsp_window.close()
        except Exception as e:
            self.show_status(f'Error context context: {e}')

    def show_status(self, msg):
        self.status_bar.setText(msg)
        
        # Safe callback cleanup optimization for threads during completions and cancellations
        if msg in ['Detection Completed', 'Detection Terminated', '檢測完成', '檢測終止']:
            if self.PageIndex == 0:
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                self.run_button.setChecked(False)
                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()
                if msg in ['Detection Terminated', '檢測終止']:
                    self.position_changed("0,1")
                    self.pre_video.clear()
                    self.res_video.clear()
                    self.time_label.setText('00:00:00 / 00:00:00')
                    self.Class_num.setText('--')
                    self.Target_num.setText('--')
                    self.fps_label.setText('--')
            elif self.PageIndex == 2:
                self.save_res_button_cam.setEnabled(True)
                self.save_txt_button_cam.setEnabled(True)
                self.run_button_cam.setChecked(False)
                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()
                if msg in ['Detection Terminated', '檢測終止']:
                    self.pre_cam.clear()
                    self.res_cam.clear()
                    self.time_label_cam.setText('Program Runtime: 00:00:00')
                    self.Class_num_cam.setText('--')
                    self.Target_num_cam.setText('--')
                    self.fps_label_cam.setText('--')

        elif msg == 'Welcome to YOLOv8 Detection System, please select a Mode':
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()
            self.position_changed("0,1")
            self.pre_video.clear()
            self.res_video.clear()
            self.pre_cam.clear()
            self.res_cam.clear()
            self.time_label.setText('00:00:00 / 00:00:00')
            self.time_label_cam.setText('Program Runtime: 00:00:00')
            for label in [self.Class_num, self.Target_num, self.fps_label, self.Class_num_cam, self.Target_num_cam, self.fps_label_cam]:
                label.setText('--')

    def ModelBoxRefre(self):
        folder_paths = {
            'Classify': './models/classify', 'Detect': './models/detect',
            'Pose': './models/pose', 'Segment': './models/segment', 'Track': './models/track'
        }
        if self.task not in folder_paths:
            return
            
        folder = folder_paths[self.task]
        if not os.path.exists(folder):
            return
            
        files = [f for f in os.listdir(folder) if f.endswith(('.pt', '.onnx', '.engine'))]
        files.sort(key=lambda x: os.path.getsize(os.path.join(folder, x)))
        
        if files != self.pt_list:
            self.pt_list = files
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)
            
            self.pt_list_cam = files.copy()
            self.model_box_cam.clear()
            self.model_box_cam.addItems(self.pt_list_cam)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPosition().toPoint()

    def resizeEvent(self, event):
        UIFunctions.resize_grips(self)
        super().resizeEvent(event)

    def load_config(self):
        config_file = 'config/setting.json'
        default_config = {
            "iou": 0.26, "conf": 0.33, "rate": 10,
            "save_res": 0, "save_txt": 0, "save_res_cam": 0, "save_txt_cam": 0
        }
        config = default_config.copy()
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config.update(json.load(f))
            except Exception:
                pass

        ui_elements = {
            "save_res": (self.save_res_button, self.yolo_predict, "save_res"),
            "save_txt": (self.save_txt_button, self.yolo_predict, "save_txt"),
            "save_res_cam": (self.save_res_button_cam, self.yolo_predict_cam, "save_res_cam"),
            "save_txt_cam": (self.save_txt_button_cam, self.yolo_predict_cam, "save_txt_cam"),
        }

        for key, (button, instance, attr) in ui_elements.items():
            button.setCheckState(Qt.Checked if config.get(key, 0) else Qt.Unchecked)
            setattr(instance, attr, config.get(key, 0) != 0)
            
        self.is_save_res()
        self.is_save_txt()
        self.cam_is_save_res()
        self.cam_is_save_txt()
        self.run_button.setChecked(False)
        self.run_button_cam.setChecked(False)

    def closeEvent(self, event):
        """[Thread sync optimization] Handles race conditions on teardown. Cleanly stops workers before closing handles."""
        config_file = 'config/setting.json'
        config = {
            "iou": self.iou_spinbox.value(),
            "conf": self.conf_spinbox.value(),
            "rate": self.speed_spinbox.value(),
            "save_res": 2 if self.save_res_button.isChecked() else 0,
            "save_txt": 2 if self.save_txt_button.isChecked() else 0,
            "save_res_cam": 2 if self.save_res_button_cam.isChecked() else 0,
            "save_txt_cam": 2 if self.save_txt_button_cam.isChecked() else 0
        }
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Request termination flags and join background processes to prevent memory dumps
        self.yolo_predict.stop_dtc = True
        if self.yolo_thread.isRunning():
            self.yolo_thread.requestInterruption()
            self.yolo_thread.quit()
            self.yolo_thread.wait(2000)  # Wait for a maximum of 2 seconds safely

        self.yolo_predict_cam.stop_dtc = True
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.requestInterruption()
            self.yolo_thread_cam.quit()
            self.yolo_thread_cam.wait(2000)

        event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            file_path = event.mimeData().urls()[0].toLocalFile()
            if not file_path:
                return
                
            if os.path.isdir(file_path):
                image_formats = {".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2", ".mp4", ".avi"}
                image_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.splitext(f)[1].lower() in image_formats]

                if image_files:
                    self.yolo_predict.source = image_files
                    self.show_status(f'Loaded Folder: {os.path.basename(file_path)}')
                    if os.path.splitext(image_files[0])[1].lower() in {".avi", ".mp4"}:
                        cap = cv2.VideoCapture(image_files[0])
                        ret, frame = cap.read()
                        if ret:
                            self.show_image(frame, self.pre_video, 'img')
                        cap.release()
                    else:
                        self.show_image(image_files[0], self.pre_video, 'path')
                else:
                    self.show_status('No supported image or video items inside selected directory...')
            else:
                self.yolo_predict.source = file_path
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext in {".avi", ".mp4"}:
                    cap = cv2.VideoCapture(file_path)
                    ret, frame = cap.read()
                    if ret:
                        self.show_image(frame, self.pre_video, 'img')
                    cap.release()
                else:
                    self.show_image(file_path, self.pre_video, 'path')

                self.show_status(f'Loaded File: {os.path.basename(file_path)}')
        except Exception as e:
            self.show_status(f'Drag-drop parsing error context: {e}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
