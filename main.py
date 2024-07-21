from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from core import YoloPredictor

from pathlib import Path
from utils.rtsp_win import Window
import traceback
import json
import sys
import cv2
import os
import numpy as np

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # 主視窗向 YOLO 實例發送執行信號
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # 基本介面設置
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 圓角透明
        self.setWindowFlags(Qt.FramelessWindowHint)  # 設置窗口標誌: 隱藏窗口邊框
        UIFuncitons.uiDefinitions(self)  # 自定義介面定義

        #初始頁面
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
        self.settings_button.setEnabled(False)

        self.src_home_button.clicked.connect(self.return_home)
        ####################################image or video####################################
        # 顯示模塊陰影
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # YOLO-v8 線程
        self.yolo_predict = YoloPredictor()                           # 創建 YOLO 實例
        self.select_model = self.model_box.currentText()                   # 默認模型
         
        self.yolo_thread = QThread()                                  # 創建 YOLO 線程
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video, 'img'))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video, 'img'))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))        
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))      
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))    
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x))) 
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        self.Qtimer_ModelBox = QTimer(self)     # 定時器: 每 2 秒監控模型文件的變化
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # 模型參數
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))    # iou 文本框
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))      # iou 滾動條
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf 文本框
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))    # conf 滾動條
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))# speed 文本框
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed 滾動條

        # 提示窗口初始化
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # 選擇資料夾來源
        self.src_file_button.clicked.connect(self.open_src_file)  # 選擇本地文件
        #單一檔案
        self.src_img_button.clicked.connect(self.open_src_img)  # 選擇本地文件
        # 開始測試按鈕
        self.run_button.clicked.connect(self.run_or_continue)   # 暫停/開始
        self.stop_button.clicked.connect(self.stop)             # 終止

        # 其他功能按鈕
        self.save_res_button.toggled.connect(self.is_save_res)  # 保存圖片選項
        self.save_txt_button.toggled.connect(self.is_save_txt)  # 保存標籤選項
        ####################################image or video####################################
        ####################################camera####################################
        # 顯示cam模塊陰影
        UIFuncitons.shadow_style(self, self.Class_QF_cam, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF_cam, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF_cam, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF_cam, QColor(64, 186, 193))

        # YOLO-v8-cam線程
        self.yolo_predict_cam = YoloPredictor()                           # 創建 YOLO 實例
        self.select_model_cam = self.model_box_cam.currentText()          # 默認模型
        
        self.yolo_thread_cam = QThread()                                  # 創建 YOLO 線程
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(lambda c: self.show_status(c))        
        self.yolo_predict_cam.yolo2main_fps.connect(lambda c: self.fps_label_cam.setText(c))      
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))    
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c))) 
        self.yolo_predict_cam.yolo2main_progress.connect(self.progress_bar_cam.setValue(0))
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

        self.Qtimer_ModelBox_cam = QTimer(self)     # 定時器: 每 2 秒監控模型文件的變化
        self.Qtimer_ModelBox_cam.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox_cam.start(2000)

        # cam模型參數
        self.model_box_cam.currentTextChanged.connect(self.cam_change_model)     
        self.iou_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_spinbox_cam'))    # iou 文本框
        self.iou_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_slider_cam'))      # iou 滾動條
        self.conf_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_spinbox_cam'))  # conf 文本框
        self.conf_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_slider_cam'))    # conf 滾動條
        self.speed_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_spinbox_cam'))# speed 文本框
        self.speed_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_slider_cam'))  # speed 滾動條

        # 提示窗口初始化
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')
        self.Model_name_cam.setText(self.select_model_cam)
        
        # 選擇檢測來源
        self.src_cam_button.clicked.connect(self.cam_button)#選擇攝像機
        
        # 開始測試按鈕
        self.run_button_cam.clicked.connect(self.cam_run_or_continue)   # 暫停/開始
        self.stop_button_cam.clicked.connect(self.cam_stop)             # 終止
        
        # 其他功能按鈕
        self.save_res_button_cam.toggled.connect(self.cam_is_save_res)  # 保存圖片選項
        self.save_txt_button_cam.toggled.connect(self.cam_is_save_txt)  # 保存標籤選項
        ####################################camera####################################
        ####################################rtsp####################################
        self.src_rtsp_button.clicked.connect(self.rtsp_button)
        ####################################rtsp####################################
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # 左側導航按鈕
        # 初始化
        self.load_config()
        self.show_status("歡迎使用YOLOv8檢測系統，請選擇Mode")

    def switch_mode(self, task):
        self.task = task
        self.yolo_predict.task = task
        self.yolo_predict_cam.task = task
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # 右上方設置按鈕

        # 讀取模型文件夾
        self.pt_list = os.listdir(f'./models/{task.lower()}/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'./models/{task.lower()}/' + x))  # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = f"./models/{task.lower()}/{self.select_model}"
        self.yolo_predict_cam.new_model_name = f"./models/{task.lower()}/{self.select_model_cam}"

        # 讀取cam模型文件夾
        self.pt_list_cam = os.listdir(f'./models/{task.lower()}/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize(f'./models/{task.lower()}/' + x))  # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status(f"目前頁面：image or video檢測頁面，Mode：{task}")

    def button_classify(self):
        self.switch_mode('Classify')

    def button_detect(self):
        self.switch_mode('Detect')

    def button_pose(self):
        self.switch_mode('Pose')

    def button_segment(self):
        self.switch_mode('Segment')

    def button_track(self):
        self.switch_mode('Track')

    def return_home(self):
        # 返回主頁面，重置狀態和按鈕
        self.PageIndex = 1
        self.content.setCurrentIndex(1)
        self.yolo_predict_cam.source = ''
        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)
        self.settings_button.setEnabled(False)
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
        self.cam_stop()
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()
        self.stop()
        self.show_status("歡迎使用YOLOv8檢測系統，請選擇Mode")

    ####################################image or video####################################
    # 選擇本地檔案
    def open_src_file(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        # 根據任務類型顯示狀態信息
        mode_status = {
            'Classify': "目前頁面：image or video檢測頁面，Mode：Classify",
            'Detect': "目前頁面：image or video檢測頁面，Mode：Detect",
            'Pose': "目前頁面：image or video檢測頁面，Mode：Pose",
            'Segment': "目前頁面：image or video檢測頁面，Mode：Segment",
            'Track': "目前頁面：image or video檢測頁面，Mode：Track"
        }
        
        # 結束cam線程，節省資源
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
            self.cam_stop()

        # 根據任務類型顯示狀態信息
        if self.task in mode_status:
            self.show_status(mode_status[self.task])
        
        # 設置配置檔路徑
        config_file = 'config/fold.json'
        
        # 讀取配置檔內容，獲取上次打開的資料夾路徑，如果不存在則使用當前工作目錄
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config.get('open_fold', os.getcwd())
        
        # 通過文件對話框讓用戶選擇資料夾
        FolderPath = QFileDialog.getExistingDirectory(self, '選擇資料夾', open_fold)
        
        # 如果用戶選擇了資料夾
        if FolderPath:
            FileFormat = [".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath) for jpgname in FileFormat
                          if jpgname in filename]
            if Foldername:
                # 將所選檔案的路徑設置為 yolo_predict 的 source
                self.yolo_predict.source = Foldername
                # 顯示檔案載入狀態
                self.show_status('載入資料夾：{}'.format(os.path.basename(FolderPath)))
                # 更新配置檔中的上次打開的資料夾路徑
                config['open_fold'] = os.path.dirname(FolderPath)
                
                # 將更新後的配置檔寫回到檔案中
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                # 停止檢測
                self.stop()
            else:
                self.show_status('資料夾內沒有圖片...')
         
    # 選擇本地檔案
    def open_src_img(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        # 根據任務類型顯示不同的狀態信息
        mode_status = {
            'Classify': "目前頁面：image or video檢測頁面，Mode：Classify",
            'Detect': "目前頁面：image or video檢測頁面，Mode：Detect",
            'Pose': "目前頁面：image or video檢測頁面，Mode：Pose",
            'Segment': "目前頁面：image or video檢測頁面，Mode：Segment",
            'Track': "目前頁面：image or video檢測頁面，Mode：Track"
        }
        if self.task in mode_status:
            self.show_status(mode_status[self.task])
        
        # 結束cam線程，節省資源
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
            self.cam_stop()

        # 設置配置檔路徑
        config_file = 'config/fold.json'
        
        # 讀取配置檔內容
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        
        # 獲取上次打開的資料夾路徑
        open_fold = config.get('open_fold', os.getcwd())
        
        # 通過文件對話框讓用戶選擇圖片或影片檔案
        if self.task == 'Track':
            title = 'Video'
            filters = "Pic File(*.mp4 *.mkv *.avi *.flv)"
        else:
            title = 'Video/image'
            filters = "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)"
        
        name, _ = QFileDialog.getOpenFileName(self, title, open_fold, filters)
        
        # 如果用戶選擇了檔案
        if name:
            # 將所選檔案的路徑設置為 yolo_predict 的 source
            self.yolo_predict.source = name
            
            # 顯示檔案載入狀態
            self.show_status('載入檔案：{}'.format(os.path.basename(name)))
            
            # 更新配置檔中的上次打開的資料夾路徑
            config['open_fold'] = os.path.dirname(name)
            
            # 將更新後的配置檔寫回到檔案中
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            
            # 停止檢測
            self.stop()
                
    # 主視窗顯示原始圖片和檢測結果
    @staticmethod
    def show_image(img_src, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img_src, dtype=np.uint8), -1)

            # 獲取原始圖片的高度、寬度和通道數
            ih, iw, _ = img_src.shape

            # 獲取標籤(label)的寬度和高度
            w, h = label.geometry().width(), label.geometry().height()

            # 保持原始數據比例，計算縮放後的尺寸
            if iw / w > ih / h:
                scal = w / iw
                nw, nh = w, int(scal * ih)
            else:
                scal = h / ih
                nw, nh = int(scal * iw), h

            # 調整圖片大小並轉換為RGB格式
            frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh)), cv2.COLOR_BGR2RGB)

            # 將圖片數據轉換為Qt的圖片對象並顯示在標籤上
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # 處理異常，印出錯誤信息
            traceback.print_exc()
            print(f"Error: {e}")
            if instance is not None:
                instance.show_status('%s' % e)

    # 控制開始/暫停檢測
    def run_or_continue(self):
        def handle_no_source():
            self.show_status('開始偵測前請選擇圖片或影片來源...')
            self.run_button.setChecked(False)

        def start_detection():
            self.save_txt_button.setEnabled(False)  # 啟動檢測後禁止勾選保存
            self.save_res_button.setEnabled(False)
            self.show_status('檢測中...')
            self.yolo_predict.continue_dtc = True  # 控制 YOLO 是否暫停

            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()

        def pause_detection():
            self.yolo_predict.continue_dtc = False
            self.show_status("檢測暫停...")
            self.run_button.setChecked(False)  # 停止按鈕

        if not self.yolo_predict.source:
            handle_no_source()
        else:
            self.yolo_predict.stop_dtc = False

            if self.run_button.isChecked():  # 如果開始按鈕被勾選
                start_detection()
            else:  # 如果開始按鈕未被勾選，表示暫停檢測
                pause_detection()

    # 保存測試結果按鈕 -- 圖片/視頻
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示運行圖片結果不會保存
            self.show_status('NOTE：運行圖片結果不會保存')
            
            # 將 YOLO 實例的保存結果的標誌設置為 False
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示運行圖片結果將會保存
            self.show_status('NOTE：運行圖片結果將會保存')
            
            # 將 YOLO 實例的保存結果的標誌設置為 True
            self.yolo_predict.save_res = True

    # 保存測試結果按鈕 -- 標籤（txt）
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示標籤結果不會保存
            self.show_status('NOTE：Label結果不會保存')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 False
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示標籤結果將會保存
            self.show_status('NOTE：Label結果將會保存')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 True
            self.yolo_predict.save_txt = True

    # 終止按鈕及相關狀態處理
    def stop(self):
        def stop_yolo_thread():
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 結束線程
            self.yolo_predict.stop_dtc = True

        def reset_ui_elements():
            self.run_button.setChecked(False)
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.pre_video.clear()
            self.res_video.clear()
            self.progress_bar.setValue(0)
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

        stop_yolo_thread()
        self.show_status('檢測終止')
        reset_ui_elements()

    # 更改檢測參數
    def change_val(self, x, flag):
        def update_iou():
            value = x / 100
            self.iou_spinbox.setValue(value)
            self.show_status(f'IOU Threshold: {value}')
            self.yolo_predict.iou_thres = value

        def update_conf():
            value = x / 100
            self.conf_spinbox.setValue(value)
            self.show_status(f'Conf Threshold: {value}')
            self.yolo_predict.conf_thres = value

        def update_speed():
            self.speed_spinbox.setValue(x)
            self.show_status(f'Delay: {x} ms')
            self.yolo_predict.speed_thres = x  # 毫秒

        update_actions = {
            'iou_spinbox': lambda: self.iou_slider.setValue(int(x * 100)),
            'iou_slider': update_iou,
            'conf_spinbox': lambda: self.conf_slider.setValue(int(x * 100)),
            'conf_slider': update_conf,
            'speed_spinbox': lambda: self.speed_slider.setValue(x),
            'speed_slider': update_speed
        }

        if flag in update_actions:
            update_actions[flag]()
    
    # 更改模型
    def change_model(self, x):
        # 獲取當前選擇的模型名稱
        self.select_model = self.model_box.currentText()
        
        # 根據任務設置模型路徑的前綴
        model_prefix = {
            'Classify': './models/classify/',
            'Detect': './models/detect/',
            'Pose': './models/pose/',
            'Segment': './models/segment/',
            'Track': './models/track/'
        }.get(self.task, './models/')

        # 設置 YOLO 實例的新模型名稱
        self.yolo_predict.new_model_name = f"{model_prefix}{self.task.lower()}/{self.select_model}"

        # 顯示消息，提示模型已更改
        self.show_status(f'Change Model: {self.select_model}')
        
        # 在界面上顯示新的模型名稱
        self.Model_name.setText(self.select_model)
    ####################################image or video####################################

    ####################################camera####################################
    def cam_button(self):
        self.yolo_predict_cam.source = 0
        self.show_status('目前頁面：Webcam檢測頁面')
        # 結束image or video線程，節省資源
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            self.yolo_thread.quit() # 結束線程
            self.yolo_thread_cam.quit()
            self.stop()
            self.cam_stop()

        if self.PageIndex != 2:
            self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))   # 右上方設置按鈕
            
    # cam控制開始/暫停檢測
    def cam_run_or_continue(self):
        def handle_no_camera():
            self.show_status('並未檢測到攝影機')
            self.run_button_cam.setChecked(False)

        def start_detection():
            self.run_button_cam.setChecked(True)  # 啟動按鈕
            self.save_txt_button_cam.setEnabled(False)  # 啟動檢測後禁止勾選保存
            self.save_res_button_cam.setEnabled(False)
            self.show_status('檢測中...')
            self.yolo_predict_cam.continue_dtc = True

            if not self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.start()
                self.main2yolo_begin_sgl.emit()

        def pause_detection():
            self.yolo_predict_cam.continue_dtc = False
            self.show_status("檢測暫停...")
            self.run_button_cam.setChecked(False)  # 停止按鈕

        if self.yolo_predict_cam.source == '':
            handle_no_camera()
        else:
            self.yolo_predict_cam.stop_dtc = False

            if self.run_button_cam.isChecked():
                start_detection()
            else:
                pause_detection()

    # cam主視窗顯示原始圖片和檢測結果
    @staticmethod
    def cam_show_image(img_src, label, instance=None):
        try:
            # 獲取原始圖片的高度、寬度和通道數
            ih, iw, _ = img_src.shape

            # 獲取標籤(label)的寬度和高度
            w, h = label.geometry().width(), label.geometry().height()

            # 保持原始數據比例，計算縮放後的尺寸
            if iw / w > ih / h:
                scal = w / iw
                nw, nh = w, int(scal * ih)
            else:
                scal = h / ih
                nw, nh = int(scal * iw), h

            # 調整圖片大小並轉換為RGB格式
            frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh)), cv2.COLOR_BGR2RGB)

            # 將圖片數據轉換為Qt的圖片對象並顯示在標籤上
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # 處理異常，印出錯誤信息
            traceback.print_exc()
            print(f"Error: {e}")
            if instance is not None:
                instance.show_status('%s' % e)

    # 更改檢測參數
    def cam_change_val(self, c, flag):
        def update_iou():
            value = c / 100
            self.iou_spinbox_cam.setValue(value)
            self.show_status(f'IOU Threshold: {value}')
            self.yolo_predict_cam.iou_thres = value

        def update_conf():
            value = c / 100
            self.conf_spinbox_cam.setValue(value)
            self.show_status(f'Conf Threshold: {value}')
            self.yolo_predict_cam.conf_thres = value

        def update_speed():
            self.speed_spinbox_cam.setValue(c)
            self.show_status(f'Delay: {c} ms')
            self.yolo_predict_cam.speed_thres = c  # 毫秒

        update_actions = {
            'iou_spinbox_cam': lambda: self.iou_slider_cam.setValue(int(c * 100)),
            'iou_slider_cam': update_iou,
            'conf_spinbox_cam': lambda: self.conf_slider_cam.setValue(int(c * 100)),
            'conf_slider_cam': update_conf,
            'speed_spinbox_cam': lambda: self.speed_spinbox_cam.setValue(c),
            'speed_slider_cam': update_speed,
        }

        if flag in update_actions:
            update_actions[flag]()

    # 更改模型
    def cam_change_model(self, c):
        # 獲取當前選擇的模型名稱
        self.select_model_cam = self.model_box_cam.currentText()
        
        # 根據任務設置模型路徑的前綴
        model_prefix = {
            'Classify': './models/classify/',
            'Detect': './models/detect/',
            'Pose': './models/pose/',
            'Segment': './models/segment/',
            'Track': './models/track/'
        }.get(self.task, './models/')

        # 設置 YOLO 實例的新模型名稱
        self.yolo_predict_cam.new_model_name = f"{model_prefix}{self.task.lower()}/{self.select_model_cam}"

        # 顯示消息，提示模型已更改
        self.show_status(f'Change Model: {self.select_model_cam}')
        
        # 在界面上顯示新的模型名稱
        self.Model_name_cam.setText(self.select_model_cam)

    # 保存測試結果按鈕 -- 圖片/視頻
    def cam_is_save_res(self):
        if self.save_res_button_cam.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示運行圖片結果不會保存
            self.show_status('NOTE：Webcam結果不會保存')
            
            # 將 YOLO 實例的保存結果的標誌設置為 False
            self.yolo_thread_cam.save_res = False
        elif self.save_res_button_cam.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示運行圖片結果將會保存
            self.show_status('NOTE：Webcam結果將會保存')
            
            # 將 YOLO 實例的保存結果的標誌設置為 True
            self.yolo_thread_cam.save_res = True

    # 保存測試結果按鈕 -- 標籤（txt）
    def cam_is_save_txt(self):
        if self.save_txt_button_cam.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示標籤結果不會保存
            self.show_status('NOTE：Label結果不會保存')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 False
            self.yolo_thread_cam.save_txt_cam = False
        elif self.save_txt_button_cam.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示標籤結果將會保存
            self.show_status('NOTE：Label結果將會保存')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 True
            self.yolo_thread_cam.save_txt_cam = True

    # cam終止按鈕及相關狀態處理
    def cam_stop(self):
        def stop_yolo_thread():
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # 結束線程
            self.yolo_predict_cam.stop_dtc = True

        def reset_ui_elements():
            self.run_button_cam.setChecked(False)
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)
            self.pre_cam.clear()
            self.res_cam.clear()
            self.Class_num_cam.setText('--')
            self.Target_num_cam.setText('--')
            self.fps_label_cam.setText('--')

        stop_yolo_thread()
        self.show_status('檢測終止')
        reset_ui_elements()
    ####################################camera####################################
    ####################################rtsp####################################
    # rtsp輸入地址
    def rtsp_button(self):
        def stop_yolo_threads():
            # 如果 YOLO 線程正在運行，則終止線程
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()

            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()

            # 停止 YOLO 實例檢測
            self.stop()
            self.cam_stop()

        def load_rtsp_window():
            self.rtsp_window = Window()
            config_file = 'config/ip.json'

            if not os.path.exists(config_file):
                ip = "rtsp://admin:admin888@192.168.1.2:555"
                new_config = {"ip": ip}
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, ensure_ascii=False, indent=2)
            else:
                config = json.load(open(config_file, 'r', encoding='utf-8'))
                ip = config['ip']

            self.rtsp_window.rtspEdit.setText(ip)
            self.rtsp_window.show()
            self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

        self.yolo_predict_cam.stream_buffer = True
        # 停止影像或視頻線程，節省資源
        stop_yolo_threads()

        # 切換至 RTSP 檢測頁面
        self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.show_status('目前頁面：RTSP 檢測頁面')

        # 加載 RTSP 設置窗口
        load_rtsp_window()

        # 設置右上角設置按鈕連接函數
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))

    # 載入網路來源
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加載 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict_cam.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)
    ####################################rtsp####################################
    ####################################共用####################################
    # 顯示底部狀態欄信息
    def show_status(self, msg):
        self.status_bar.setText(msg)
        def handle_page_0():
            if msg == '檢測完成':
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                self.run_button.setChecked(False)

                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()

            elif msg == '檢測終止':
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                self.run_button.setChecked(False)
                self.progress_bar.setValue(0)

                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()

                self.pre_video.clear()
                self.res_video.clear()
                self.Class_num.setText('--')
                self.Target_num.setText('--')
                self.fps_label.setText('--')

        def handle_page_2():
            if msg == '檢測終止':
                self.save_res_button_cam.setEnabled(True)
                self.save_txt_button_cam.setEnabled(True)
                self.run_button_cam.setChecked(False)
                self.progress_bar_cam.setValue(0)

                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()

                self.pre_cam.clear()
                self.res_cam.clear()
                self.Class_num_cam.setText('--')
                self.Target_num_cam.setText('--')
                self.fps_label_cam.setText('--')

        # 根據不同的頁面處理不同的狀態
        if self.PageIndex == 0:
            handle_page_0()
        elif self.PageIndex == 2:
            handle_page_2()

    # 循環監控模型文件更改
    def ModelBoxRefre(self):
        def update_model_box(folder):
            pt_list = os.listdir(folder)
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize(os.path.join(folder, x)))
            return pt_list

        folder_paths = {
            'Classify': './models/classify',
            'Detect': './models/detect',
            'Pose': './models/pose',
            'Segment': './models/segment',
            'Track': './models/track'
        }

        if self.task in folder_paths:
            pt_list = update_model_box(folder_paths[self.task])
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

    # 獲取滑鼠位置（用於按住標題欄拖動窗口）
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # 在調整窗口大小時進行優化調整（針對拖動窗口右下角邊緣調整窗口大小）
    def resizeEvent(self, event):
        # 更新大小調整的手柄
        UIFuncitons.resize_grips(self)

    # 配置初始化
    def load_config(self):
        config_file = 'config/setting.json'
        
        default_config = {
            "iou": 0.26,
            "conf": 0.33,
            "rate": 10,
            "save_res": 0,
            "save_txt": 0,
            "save_res_cam": 0,
            "save_txt_cam": 0
        }

        config = default_config.copy()
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        
        def update_ui(config):
            ui_elements = {
                "save_res": (self.save_res_button, self.yolo_predict, "save_res"),
                "save_txt": (self.save_txt_button, self.yolo_predict, "save_txt"),
                "save_res_cam": (self.save_res_button_cam, self.yolo_predict_cam, "save_res_cam"),
                "save_txt_cam": (self.save_txt_button_cam, self.yolo_predict_cam, "save_txt_cam"),
            }

            for key, (button, instance, attr) in ui_elements.items():
                button.setCheckState(Qt.Checked if config[key] else Qt.Unchecked)
                setattr(instance, attr, config[key] != 0)

            self.run_button.setChecked(False)
            self.run_button_cam.setChecked(False)
        
        update_ui(config)

    # 關閉事件，退出線程，保存設置
    def closeEvent(self, event):
        # 保存配置到設定文件
        config_file = 'config/setting.json'
        config = {
            "iou": self.iou_spinbox.value(),
            "conf": self.conf_spinbox.value(),
            "rate": self.speed_spinbox.value(),
            "save_res": 0 if self.save_res_button.checkState() == Qt.Unchecked else 2,
            "save_txt": 0 if self.save_txt_button.checkState() == Qt.Unchecked else 2,
            "save_res_cam": 0 if self.save_res_button_cam.checkState() == Qt.Unchecked else 2,
            "save_txt_cam": 0 if self.save_txt_button_cam.checkState() == Qt.Unchecked else 2
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 退出線程和應用程序
        def quit_threads():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()

            self.yolo_predict_cam.stop_dtc = True
            self.yolo_thread_cam.quit()
            
            # 顯示退出提示，等待3秒
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            
            # 退出應用程序
            sys.exit(0)
        
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            quit_threads()
        else:
            sys.exit(0)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        
    def dropEvent(self, event):
        def handle_directory(directory):
            image_formats = {".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"}
            image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.splitext(filename)[1].lower() in image_formats]

            if image_files:
                self.yolo_predict.source = image_files
                self.show_image(self.yolo_predict.source[0], self.pre_video, 'path')
                self.show_status('載入資料夾：{}'.format(os.path.basename(directory)))
            else:
                self.show_status('資料夾內沒有圖片...')

        def handle_file(file):
            self.yolo_predict.source = file
            file_ext = os.path.splitext(file)[1].lower()

            if file_ext in {".avi", ".mp4"}:
                self.cap = cv2.VideoCapture(self.yolo_predict.source)
                ret, frame = self.cap.read()
                if ret:
                    self.show_image(frame, self.pre_video, 'img')
            else:
                self.show_image(self.yolo_predict.source, self.pre_video, 'path')

            self.show_status('載入檔案：{}'.format(os.path.basename(self.yolo_predict.source)))

        try:
            file = event.mimeData().urls()[0].toLocalFile()
            if file:
                if os.path.isdir(file):
                    handle_directory(file)
                else:
                    handle_file(file)
        except Exception as e:
            self.show_status('錯誤：{}'.format(e))
    ####################################共用####################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
