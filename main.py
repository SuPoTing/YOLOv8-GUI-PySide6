####################ultralytics==8.0.229####################
####################cuda11.8####################
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imgsz, check_imshow, check_yaml
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.trackers import track
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from PIL import Image
from utils.capnums import Camera
from utils.rtsp_win import Window
import numpy as np
import threading
import traceback
import time
import json
import torch
import sys
import cv2
import os

class CameraThread(QThread):
    imageCaptured = Signal(np.ndarray)

    def __init__(self, parent=None):
        super(CameraThread, self).__init__(parent)
        self.camera_source = 0  # 設置相機源，這裡使用默認相機

    def run(self):
        cap = cv2.VideoCapture(self.camera_source)

        while (cap.isOpened()):
            if self.isInterruptionRequested():
                break

            ret, frame = cap.read()
            if ret:
                self.imageCaptured.emit(frame)

            else:
                self.imageCaptured.emit(np.array([]))# 快完成了

        cap.release()      




class YoloPredictor(BasePredictor, QObject):
    # 信號定義，用於與其他部分進行通信
    yolo2main_pre_img = Signal(np.ndarray)   # 原始圖像信號
    yolo2main_res_img = Signal(np.ndarray)   # 測試結果信號
    yolo2main_status_msg = Signal(str)       # 檢測/暫停/停止/測試完成/錯誤報告信號
    yolo2main_fps = Signal(str)              # 幀率信號
    yolo2main_labels = Signal(dict)          # 檢測目標結果（每個類別的數量）
    yolo2main_progress = Signal(int)         # 完整度信號
    yolo2main_class_num = Signal(int)        # 檢測到的類別數量
    yolo2main_target_num = Signal(int)       # 檢測到的目標數量

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # 調用父類的初始化方法
        super(YoloPredictor, self).__init__()
        # 初始化 PyQt 的 QObject
        QObject.__init__(self)

        # 解析配置文件
        self.args = get_cfg(cfg, overrides)
        # 設置模型保存目錄
        self.save_dir = get_save_dir(self.args)
        # 初始化一個標誌，標記模型是否已經完成預熱（warmup）
        self.done_warmup = False
        # 檢查是否要顯示圖像
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI 相關的屬性
        self.used_model_name = None  # 要使用的檢測模型的名稱
        self.new_model_name = None   # 實時更改的模型名稱
        self.source = ''             # 輸入源
        self.stop_dtc = False        # 終止檢測的標誌
        self.continue_dtc = True     # 暫停檢測的標誌
        self.save_res = False        # 保存測試結果的標誌
        self.save_txt = False        # 保存標籤（txt）文件的標誌
        self.iou_thres = 0.45        # IoU 閾值
        self.conf_thres = 0.25       # 置信度閾值
        self.speed_thres = 0         # 延遲，毫秒
        self.labels_dict = {}        # 返回檢測結果的字典
        self.progress_value = 0      # 進度條的值
        self.task = ''

        # 如果設置已完成，可以使用以下屬性
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    # main for detect
    @smart_inference_mode()
    def run(self, *args, **kwargs):
        try:
            if self.args.verbose:
                LOGGER.info('')
            # Setup model
            self.yolo2main_status_msg.emit('模型載入中...')
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            with self._lock:  # for thread-safe inference
                # Setup source every time predict is called
                self.setup_source(self.source if self.source is not None else self.args.source)

                # 檢查保存路徑/標籤
                if self.save_res or self.save_txt:
                    (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

                # 模型預熱
                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                    self.done_warmup = True

                self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
                # 開始檢測
                count = 0                       # frame count
                start_time = time.time()        # 用於計算幀率
                # batch = iter(self.dataset)
                for batch in self.dataset:
                # while True:
                    # 終止檢測標誌檢測
                    if self.stop_dtc:
                        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                            self.vid_writer[-1].release()  # 釋放最後的視訊寫入器
                        self.yolo2main_status_msg.emit('檢測終止')
                        break
                    
                    # 在中途更改模型
                    if self.used_model_name != self.new_model_name:  
                        # self.yolo2main_status_msg.emit('Change Model...')
                        self.setup_model(self.new_model_name)
                        self.used_model_name = self.new_model_name
                    
                    # 暫停開關
                    if self.continue_dtc:
                        # time.sleep(0.001)
                        self.yolo2main_status_msg.emit('檢測中...')
                        # batch = next(self.dataset)  # 獲取下一個數據

                        self.batch = batch
                        path, im0s, vid_cap, s = batch
                        visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

                        # 計算完成度和幀率（待優化）
                        count += 1              # 幀計數 +1
                        if vid_cap:
                            all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)   # 總幀數
                        else:
                            all_count = 1
                        self.progress_value = int(count/all_count*1000)         # 進度條（0~1000）
                        if count % 5 == 0 and count >= 5:                     # 每5幀計算一次幀率
                            self.yolo2main_fps.emit(str(int(5/(time.time()-start_time))))
                            start_time = time.time()
                        # Preprocess
                        with profilers[0]:
                            if self.task == 'Classify':
                                self._legacy_transform_name = 'ultralytics.yolo.data.augment.ToTensor'
                                im = self.classify_preprocess(im0s)
                            else:
                                im = self.preprocess(im0s)
                            # elif self.task == 'Detect' or self.task == 'Pose' or self.task == 'Segment':    
                            #     im = self.preprocess(im0s)

                        # Inference
                        with profilers[1]:
                            preds = self.inference(im, *args, **kwargs)
                            # if self.args.embed:
                                # yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                                # continue

                        # Postprocess
                        with profilers[2]:
                            if self.task == 'Classify':
                                self.results = self.classify_postprocess(preds, im, im0s)
                            elif self.task == 'Detect':
                                self.results = self.postprocess(preds, im, im0s)
                            elif self.task == 'Pose':
                                self.results = self.pose_postprocess(preds, im, im0s)
                            elif self.task == 'Segment':
                                self.results = self.segment_postprocess(preds, im, im0s)

                            elif self.task == 'Track':
                                model = YOLO(self.used_model_name)
                                self.results = model.track(source=self.source, tracker="bytetrack.yaml")
                                
                                # pass
                        self.run_callbacks('on_predict_postprocess_end')
                        # Visualize, save, write results
                        n = len(im0s)
                        for i in range(n):
                            self.seen += 1
                            self.results[i].speed = {
                                'preprocess': profilers[0].dt * 1E3 / n,
                                'inference': profilers[1].dt * 1E3 / n,
                                'postprocess': profilers[2].dt * 1E3 / n}
                            p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                            p = Path(p)
                            label_str = self.write_results(i, self.results, (p, im, im0))
                            # 標籤和數字字典
                            class_nums = 0
                            target_nums = 0
                            self.labels_dict = {}
                            if 'no detections' in label_str:
                                im0 = im0
                                pass
                            else:
                                im0 = self.plotted_img
                                for ii in label_str.split(',')[:-1]:
                                    nums, label_name = ii.split('~')
                                    if ':' in nums:
                                        _, nums = nums.split(':')
                                    self.labels_dict[label_name] = int(nums)
                                    target_nums += int(nums)
                                    class_nums += 1
                            if self.save_res:
                                self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                            # 發送測試結果
                            self.yolo2main_res_img.emit(im0) # 檢測後
                            self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])   # 檢測前
                            # self.yolo2main_labels.emit(self.labels_dict)        # webcam 需要更改 def write_results
                            self.yolo2main_class_num.emit(class_nums)
                            self.yolo2main_target_num.emit(target_nums)

                            if self.speed_thres != 0:
                                time.sleep(self.speed_thres/1000)   # 延遲，毫秒

                        self.yolo2main_progress.emit(self.progress_value)   # 進度條


                    # 檢測完成
                    if not self.source_type.webcam and count + 1 >= all_count:
                        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                            self.vid_writer[-1].release()  # 釋放最後的視頻寫入器
                        self.yolo2main_status_msg.emit('檢測完成')
                        break

        except Exception as e:
            pass
            traceback.print_exc()
            print(f"Error: {e}")
            self.yolo2main_status_msg.emit('%s' % e)

    def inference(self, img, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(img, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def classify_preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(self._legacy_transform_name in str(transform)
                                      for transform in self.transforms.transforms)
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack([self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img],
                                  dim=0)
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32


    def classify_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, probs=pred))
        return results

    def preprocess(self, img):
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        ### important
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pose_postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts))
        return results

    def segment_postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(preds[0],
                                    self.conf_thres,
                                    self.iou_thres,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def pre_transform(self, img):
        same_shapes = all(x.shape == img[0].shape for x in img)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in img]

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.task == 'Classify' else None
        self.dataset = load_inference_source(source=source,
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             buffer=self.args.stream_buffer)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        # log_string += '%gx%g ' % im.shape[2:]  # print string

        result = results[idx]
        
        if self.task == 'Classify':
            prob = results[idx].probs
            print(prob)
            # for c in prob.top5:
            #     print(c)
        else:
            det = results[idx].boxes
            if len(det) == 0:
                return f'{log_string}(no detections), ' # if no, send this~~

            for c in det.cls.unique():
                n = (det.cls == c).sum()  # detections per class
                log_string += f"{n}~{self.model.names[int(c)]},"

        if self.save_res or self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.show_boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops',
                             file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))


        return log_string

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
        self.pushButton_track.setEnabled(False)
        
        self.src_file_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)
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
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
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

        # 選擇檢測來源
        self.src_file_button.clicked.connect(self.open_src_file)  # 選擇本地文件
        self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#選擇 RTSP

        # 開始測試按鈕
        self.run_button.clicked.connect(self.run_or_continue)   # 暫停/開始
        self.stop_button.clicked.connect(self.stop)             # 終止

        # 其他功能按鈕
        self.save_res_button.toggled.connect(self.is_save_res)  # 保存圖片選項
        self.save_txt_button.toggled.connect(self.is_save_txt)  # 保存標籤選項
        ####################################image or video####################################

        ####################################camera####################################
        self.cam_data = np.array([])
        # 顯示cam模塊陰影
        UIFuncitons.shadow_style(self, self.Class_QF_cam, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF_cam, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF_cam, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF_cam, QColor(64, 186, 193))

        # YOLO-v8-cam線程
        self.yolo_predict_cam = YoloPredictor()                           # 創建 YOLO 實例
        self.select_model_cam = self.model_box_cam.currentText()                   # 默認模型
        
        self.yolo_thread_cam = QThread()                                  # 創建 YOLO 線程
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(lambda c: self.cam_show_status(c))        
        self.yolo_predict_cam.yolo2main_fps.connect(lambda c: self.fps_label_cam.setText(c))      
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))    
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c))) 
        # self.yolo_predict_cam.yolo2main_progress.connect(lambda c: self.progress_bar_cam.setValue(c))
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

        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # 左側導航按鈕
       
        # 初始化
        self.load_config()

    def button_classify(self): #觸發button_detect後的事件
        self.task = 'Classify'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task

        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # 右上方設置按鈕

        # 讀取模型文件夾
        self.pt_list = os.listdir('./models/classify/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/classify/' + x))   # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/classify/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/classify/%s" % self.select_model_cam

        # 讀取cam模型文件夾
        self.pt_list_cam = os.listdir('./models/classify/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/classify/' + x))   # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("目前頁面：image or video檢測頁面，Mode：Classify")

    def button_detect(self): #觸發button_detect後的事件
        self.task = 'Detect'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/detect/%s" % self.select_model_cam
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # 右上方設置按鈕

        # 讀取模型文件夾
        self.pt_list = os.listdir('./models/detect/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/detect/' + x))   # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/detect/%s" % self.select_model_cam

        # 讀取cam模型文件夾
        self.pt_list_cam = os.listdir('./models/detect/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/detect/' + x))   # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("目前頁面：image or video檢測頁面，Mode：Detect")

    def button_pose(self): #觸發button_detect後的事件
        self.task = 'Pose'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/pose/%s" % self.select_model_cam
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # 右上方設置按鈕

        # 讀取模型文件夾
        self.pt_list = os.listdir('./models/pose/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/pose/' + x))   # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/pose/%s" % self.select_model_cam

        # 讀取cam模型文件夾
        self.pt_list_cam = os.listdir('./models/pose/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/pose/' + x))   # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("目前頁面：image or video檢測頁面，Mode：Pose")

    def button_segment(self): #觸發button_detect後的事件
        self.task = 'Segment'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/segment/%s" % self.select_model_cam
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # 右上方設置按鈕

        # 讀取模型文件夾
        self.pt_list = os.listdir('./models/segment/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/segment/' + x))   # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/segment/%s" % self.select_model_cam

        # 讀取cam模型文件夾
        self.pt_list_cam = os.listdir('./models/segment/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/segment/' + x))   # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("目前頁面：image or video檢測頁面，Mode：Segment")

    def button_track(self): #觸發button_detect後的事件
        self.task = 'Track'
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/track/%s" % self.select_model_cam
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # 右上方設置按鈕

        # 讀取模型文件夾
        self.pt_list = os.listdir('./models/track/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/track/' + x))   # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = "./models/track/%s" % self.select_model_cam

        # 讀取cam模型文件夾
        self.pt_list_cam = os.listdir('./models/track/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize('./models/track/' + x))   # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("目前頁面：image or video檢測頁面，Mode：Track")

    ####################################image or video####################################
    # 選擇本地檔案
    def open_src_file(self):
        if self.task == 'Classify':
            self.show_status("目前頁面：image or video檢測頁面，Mode：Classify")
        if self.task == 'Detect':
            self.show_status("目前頁面：image or video檢測頁面，Mode：Detect")
        if self.task == 'Pose':
            self.show_status("目前頁面：image or video檢測頁面，Mode：Pose")
        if self.task == 'Segment':
            self.show_status("目前頁面：image or video檢測頁面，Mode：Segment")
        if self.task == 'Track':
            self.show_status("目前頁面：image or video檢測頁面，Mode：Track")      
            
        # 結束cam線程，節省資源
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit() # 結束線程
            self.cam_stop()
        # 0:image/video page
        # 1:home page
        # 2:camera page
        if self.PageIndex != 0:
            self.PageIndex = 0
            self.content.setCurrentIndex(self.PageIndex)
            self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # 右上方設置按鈕

        if self.PageIndex == 0:
            # 設置配置檔路徑
            config_file = 'config/fold.json'
            
            # 讀取配置檔內容
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            
            # 獲取上次打開的資料夾路徑
            open_fold = config['open_fold']
            
            # 如果上次打開的資料夾不存在，則使用當前工作目錄
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
            
            # 通過文件對話框讓用戶選擇圖片或影片檔案
            if self.task == 'Track':
                name, _ = QFileDialog.getOpenFileName(self, 'Video', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv)")
            else:
                name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
            
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
    def show_image(img_src, label):
        try:
            # 獲取原始圖片的高度、寬度和通道數
            ih, iw, _ = img_src.shape
            
            # 獲取標籤(label)的寬度和高度
            w = label.geometry().width()
            h = label.geometry().height()
            
            # 保持原始數據比例
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

            # 將圖片轉換為RGB格式
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            
            # 將圖片數據轉換為Qt的圖片對象
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            
            # 將圖片顯示在標籤(label)上
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # 處理異常，印出錯誤信息
            print(repr(e))

    # 控制開始/暫停檢測
    def run_or_continue(self):
        # 檢查 YOLO 預測的來源是否為空
        if self.yolo_predict.source == '':
            self.show_status('開始偵測前請選擇圖片或影片來源...')
            self.run_button.setChecked(False)
        else:
            # 設置 YOLO 預測的停止標誌為 False
            self.yolo_predict.stop_dtc = False
            
            # 如果開始按鈕被勾選
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # 啟動按鈕
                self.save_txt_button.setEnabled(False)  # 啟動檢測後禁止勾選保存
                self.save_res_button.setEnabled(False)
                self.show_status('檢測中...')           
                self.yolo_predict.continue_dtc = True   # 控制 YOLO 是否暫停
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            # 如果開始按鈕未被勾選，表示暫停檢測
            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("檢測暫停...")
                self.run_button.setChecked(False)  # 停止按鈕

    # 顯示底部狀態欄信息
    def show_status(self, msg):
        # 設置狀態欄文字
        self.status_bar.setText(msg)
        
        # 根據不同的狀態信息執行相應的操作
        if msg == 'Detection completed' or msg == '檢測完成':
            # 啟用保存結果和保存文本的按鈕
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            
            # 將檢測開關按鈕設置為未勾選狀態
            self.run_button.setChecked(False)    
            
            # 將進度條的值設置為0
            self.progress_bar.setValue(0)
            
            # 如果 YOLO 線程正在運行，則終止該線程
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 結束處理

        elif msg == 'Detection terminated!' or msg == '檢測終止':
            # 啟用保存結果和保存文本的按鈕
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            
            # 將檢測開關按鈕設置為未勾選狀態
            self.run_button.setChecked(False)    
            
            # 將進度條的值設置為0
            self.progress_bar.setValue(0)
            
            # 如果 YOLO 線程正在運行，則終止該線程
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 結束處理
            
            # 清空影像顯示
            self.pre_video.clear()  # 清除原始圖像
            self.res_video.clear()  # 清除檢測結果圖像
            self.Class_num.setText('--')  # 顯示的類別數目
            self.Target_num.setText('--')  # 顯示的目標數目
            self.fps_label.setText('--')  # 顯示的幀率信息

    # 保存測試結果按鈕 -- 圖片/視頻
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示運行圖片結果不會保存
            self.show_status('NOTE: Run image results are not saved.')
            
            # 將 YOLO 實例的保存結果的標誌設置為 False
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示運行圖片結果將會保存
            self.show_status('NOTE: Run image results will be saved.')
            
            # 將 YOLO 實例的保存結果的標誌設置為 True
            self.yolo_predict.save_res = True

    # 保存測試結果按鈕 -- 標籤（txt）
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示標籤結果不會保存
            self.show_status('NOTE: Labels results are not saved.')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 False
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示標籤結果將會保存
            self.show_status('NOTE: Labels results will be saved.')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 True
            self.yolo_predict.save_txt = True

    # 配置初始化
    def load_config(self):
        config_file = 'config/setting.json'
        
        # 如果配置文件不存在，則創建並寫入默認配置
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33   
            rate = 10
            save_res = 0   
            save_txt = 0    
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            # 如果配置文件存在，讀取配置
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            
            # 檢查配置內容是否完整，如果不完整，使用默認值
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        
        # 根據配置設置界面元素的狀態
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)
        self.show_status("歡迎使用YOLOv8檢測系統，請選擇Mode")
        # self.show_status("目前為image or video檢測頁面")

    # 終止按鈕及相關狀態處理
    def stop(self):
        # 如果 YOLO 線程正在運行，則終止線程
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit() # 結束線程
        
        # 設置 YOLO 實例的終止標誌為 True
        self.yolo_predict.stop_dtc = True
        
        # 恢復開始按鈕的狀態
        self.run_button.setChecked(False)  
        
        # 啟用保存按鈕的使用權限
        if self.task == 'Classify': 
            self.save_res_button.setEnabled(False)   
            self.save_txt_button.setEnabled(False)
        else:
            self.save_res_button.setEnabled(True)   
            self.save_txt_button.setEnabled(True)

        # 清空預測結果顯示區域的影象
        self.pre_video.clear()           
        
        # 清空檢測結果顯示區域的影象
        self.res_video.clear()           
        
        # 將進度條的值設置為0
        self.progress_bar.setValue(0)
        
        # 重置類別數量、目標數量和fps標籤
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # 更改檢測參數
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            # 如果是 iou_spinbox 的值發生變化，則改變 iou_slider 的值
            self.iou_slider.setValue(int(x * 100))

        elif flag == 'iou_slider':
            # 如果是 iou_slider 的值發生變化，則改變 iou_spinbox 的值
            self.iou_spinbox.setValue(x / 100)
            # 顯示消息，提示 IOU 閾值變化
            self.show_status('IOU Threshold: %s' % str(x / 100))
            # 設置 YOLO 實例的 IOU 閾值
            self.yolo_predict.iou_thres = x / 100

        elif flag == 'conf_spinbox':
            # 如果是 conf_spinbox 的值發生變化，則改變 conf_slider 的值
            self.conf_slider.setValue(int(x * 100))

        elif flag == 'conf_slider':
            # 如果是 conf_slider 的值發生變化，則改變 conf_spinbox 的值
            self.conf_spinbox.setValue(x / 100)
            # 顯示消息，提示 Confidence 閾值變化
            self.show_status('Conf Threshold: %s' % str(x / 100))
            # 設置 YOLO 實例的 Confidence 閾值
            self.yolo_predict.conf_thres = x / 100

        elif flag == 'speed_spinbox':
            # 如果是 speed_spinbox 的值發生變化，則改變 speed_slider 的值
            self.speed_slider.setValue(x)

        elif flag == 'speed_slider':
            # 如果是 speed_slider 的值發生變化，則改變 speed_spinbox 的值
            self.speed_spinbox.setValue(x)
            # 顯示消息，提示延遲時間變化
            self.show_status('Delay: %s ms' % str(x))
            # 設置 YOLO 實例的延遲時間閾值
            self.yolo_predict.speed_thres = x  # 毫秒
    
    # 更改模型
    def change_model(self, x):
        # 獲取當前選擇的模型名稱
        self.select_model = self.model_box.currentText()
        
        # 設置 YOLO 實例的新模型名稱
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
        # 顯示消息，提示模型已更改
        self.show_status('Change Model：%s' % self.select_model)
        
        # 在界面上顯示新的模型名稱
        self.Model_name.setText(self.select_model)
    ####################################image or video####################################

    ####################################camera####################################
    def cam_button(self):
        self.yolo_predict_cam.source = 0
        self.show_status('目前頁面：Webcam檢測頁面')
        # 結束image or video線程，節省資源
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit() # 結束線程
            self.stop()

        if self.PageIndex != 2:
            self.PageIndex = 2
            self.content.setCurrentIndex(self.PageIndex)
            self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))   # 右上方設置按鈕
            
    # cam控制開始/暫停檢測
    def cam_run_or_continue(self):
        if self.yolo_predict_cam.source == '':
            self.show_status('並未檢測到攝影機')
            self.run_button_cam.setChecked(False)

        else:
            # 設置 YOLO 預測的停止標誌為 False
            self.yolo_predict_cam.stop_dtc = False
            
        
            # 如果開始按鈕被勾選
            if self.run_button_cam.isChecked():
                self.run_button_cam.setChecked(True)  # 啟動按鈕
                self.save_txt_button_cam.setEnabled(False)  # 啟動檢測後禁止勾選保存
                self.save_res_button_cam.setEnabled(False)
                self.cam_show_status('檢測中...')           
                self.yolo_predict_cam.continue_dtc = True   # 控制 YOLO 是否暫停

                if not self.yolo_thread_cam.isRunning():                
                    self.yolo_thread_cam.start()
                    self.main2yolo_begin_sgl.emit()

            # 如果開始按鈕未被勾選，表示暫停檢測
            else:
                self.yolo_predict_cam.continue_dtc = False
                self.cam_show_status("檢測暫停...")
                self.run_button_cam.setChecked(False)  # 停止按鈕

    # cam主視窗顯示原始圖片和檢測結果
    @staticmethod
    def cam_show_image(img_src, label):
        try:
            # 獲取原始圖片的高度、寬度和通道數
            ih, iw, _ = img_src.shape
            
            # 獲取標籤(label)的寬度和高度
            w = label.geometry().width()
            h = label.geometry().height()
            
            # 保持原始數據比例
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

            # 將圖片轉換為RGB格式
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            
            # 將圖片數據轉換為Qt的圖片對象
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            
            # 將圖片顯示在標籤(label)上
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # 處理異常，印出錯誤信息
            traceback.print_exc()
            print(f"Error: {e}")
            self.cam_show_status('%s' % e)

    # 更改檢測參數
    def cam_change_val(self, c, flag):
        if flag == 'iou_spinbox_cam':
            # 如果是 iou_spinbox 的值發生變化，則改變 iou_slider 的值
            self.iou_slider_cam.setValue(int(c * 100))

        elif flag == 'iou_slider_cam':
            # 如果是 iou_slider 的值發生變化，則改變 iou_spinbox 的值
            self.iou_spinbox_cam.setValue(c / 100)
            # 顯示消息，提示 IOU 閾值變化
            self.cam_show_status('IOU Threshold: %s' % str(c / 100))
            # 設置 YOLO 實例的 IOU 閾值
            self.yolo_predict_cam.iou_thres = c / 100

        elif flag == 'conf_spinbox_cam':
            # 如果是 conf_spinbox 的值發生變化，則改變 conf_slider 的值
            self.conf_slider_cam.setValue(int(c * 100))

        elif flag == 'conf_slider_cam':
            # 如果是 conf_slider 的值發生變化，則改變 conf_spinbox 的值
            self.conf_spinbox_cam.setValue(c / 100)
            # 顯示消息，提示 Confidence 閾值變化
            self.cam_show_status('Conf Threshold: %s' % str(c / 100))
            # 設置 YOLO 實例的 Confidence 閾值
            self.yolo_predict_cam.conf_thres = c / 100

        elif flag == 'speed_spinbox_cam':
            # 如果是 speed_spinbox 的值發生變化，則改變 speed_slider 的值
            self.speed_slider_cam.setValue(c)

        elif flag == 'speed_slider_cam':
            # 如果是 speed_slider 的值發生變化，則改變 speed_spinbox 的值
            self.speed_spinbox_cam.setValue(c)
            # 顯示消息，提示延遲時間變化
            self.cam_show_status('Delay: %s ms' % str(c))
            # 設置 YOLO 實例的延遲時間閾值
            self.yolo_predict_cam.speed_thres = c  # 毫秒

    # 更改模型
    def cam_change_model(self, c):
        # 獲取當前選擇的模型名稱
        self.select_model_cam = self.model_box_cam.currentText()
        
        # 設置 YOLO 實例的新模型名稱
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
        # 顯示消息，提示模型已更改
        self.cam_show_status('Change Model：%s' % self.select_model_cam)
        
        # 在界面上顯示新的模型名稱
        self.Model_name_cam.setText(self.select_model_cam)

    # 顯示底部狀態欄信息
    def cam_show_status(self, msg):
        # 設置狀態欄文字
        self.status_bar.setText(msg)
        
        # 根據不同的狀態信息執行相應的操作
        if msg == 'Detection completed' or msg == '檢測完成':
            # 啟用保存結果和保存文本的按鈕
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)
            
            # 將檢測開關按鈕設置為未勾選狀態
            self.run_button_cam.setChecked(False)    
            
            # 將進度條的值設置為0
            self.progress_bar_cam.setValue(0)
            
            # 如果 YOLO 線程正在運行，則終止該線程
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # 結束處理

        elif msg == 'Detection terminated!' or msg == '檢測終止':
            # 啟用保存結果和保存文本的按鈕
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)
            
            # 將檢測開關按鈕設置為未勾選狀態
            self.run_button_cam.setChecked(False)    
            
            # 將進度條的值設置為0
            self.progress_bar_cam.setValue(0)
            
            # 如果 YOLO 線程正在運行，則終止該線程
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # 結束處理

            # 清空影像顯示
            self.pre_cam.clear()  # 清除原始圖像
            self.res_cam.clear()  # 清除檢測結果圖像
            self.Class_num_cam.setText('--')  # 顯示的類別數目
            self.Target_num_cam.setText('--')  # 顯示的目標數目
            self.fps_label_cam.setText('--')  # 顯示的幀率信息

    # 保存測試結果按鈕 -- 圖片/視頻
    def cam_is_save_res(self):
        if self.save_res_button_cam.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示運行圖片結果不會保存
            self.show_status('NOTE：運行圖片結果不會保存')
            
            # 將 YOLO 實例的保存結果的標誌設置為 False
            self.yolo_thread_cam.save_res = False
        elif self.save_res_button_cam.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示運行圖片結果將會保存
            self.show_status('NOTE：運行圖片結果將會保存')
            
            # 將 YOLO 實例的保存結果的標誌設置為 True
            self.yolo_thread_cam.save_res = True

    # 保存測試結果按鈕 -- 標籤（txt）
    def cam_is_save_txt(self):
        if self.save_txt_button_cam.checkState() == Qt.CheckState.Unchecked:
            # 顯示消息，提示標籤結果不會保存
            self.show_status('NOTE：Label結果不會保存')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 False
            self.yolo_thread_cam.save_txt = False
        elif self.save_txt_button_cam.checkState() == Qt.CheckState.Checked:
            # 顯示消息，提示標籤結果將會保存
            self.show_status('NOTE：Label結果將會保存')
            
            # 將 YOLO 實例的保存標籤的標誌設置為 True
            self.yolo_thread_cam.save_txt = True


    # cam終止按鈕及相關狀態處理
    def cam_stop(self):
        # 如果 YOLO 線程正在運行，則終止線程
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit() # 結束線程

        # 設置 YOLO 實例的終止標誌為 True
        self.yolo_predict_cam.stop_dtc = True
        
        # 恢復開始按鈕的狀態
        self.run_button_cam.setChecked(False)  

        # 啟用保存按鈕的使用權限
        if self.task == 'Classify': 
            self.save_res_button_cam.setEnabled(False)   
            self.save_txt_button_cam.setEnabled(False)
        else:
            self.save_res_button_cam.setEnabled(True)   
            self.save_txt_button_cam.setEnabled(True)
        
        # 清空預測結果顯示區域的影象
        self.pre_cam.clear()           
        
        # 清空檢測結果顯示區域的影象
        self.res_cam.clear()           
        
        # 將進度條的值設置為0
        # self.progress_bar.setValue(0)
        
        # 重置類別數量、目標數量和fps標籤
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')
    ####################################camera####################################

    ####################################共用####################################
    # 循環監控模型文件更改
    def ModelBoxRefre(self):
        # 獲取模型文件夾下的所有模型文件
        if self.task == 'Classify':
            pt_list = os.listdir('./models/classify')
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize('./models/classify/' + x))

            # 如果模型文件列表發生變化，則更新模型下拉框的內容
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        elif self.task == 'Detect':
            pt_list = os.listdir('./models/detect')
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize('./models/detect/' + x))
            # 如果模型文件列表發生變化，則更新模型下拉框的內容
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        elif self.task == 'Pose':
            pt_list = os.listdir('./models/pose')
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize('./models/pose/' + x))

            # 如果模型文件列表發生變化，則更新模型下拉框的內容
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        elif self.task == 'Segment':
            pt_list = os.listdir('./models/segment')
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize('./models/segment/' + x))

            # 如果模型文件列表發生變化，則更新模型下拉框的內容
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        elif self.task == 'Track':
            pt_list = os.listdir('./models/track')
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize('./models/track/' + x))

            # 如果模型文件列表發生變化，則更新模型下拉框的內容
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

    # 關閉事件，退出線程，保存設置
    def closeEvent(self, event):
        # 保存配置到設定文件
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        
        # 退出線程和應用程序
        if self.yolo_thread.isRunning():
            # 如果 YOLO 線程正在運行，則終止線程
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            
            # 顯示退出提示，等待3秒
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            
            # 退出應用程序
            sys.exit(0)
        else:
            # 如果 YOLO 線程未運行，直接退出應用程序
            sys.exit(0)
    ####################################共用####################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    # 創建相機線程
    # camera_thread = CameraThread()
    # camera_thread.imageCaptured.connect(Home.cam_data)
    # camera_thread.start()
    Home.show()
    sys.exit(app.exec())  
