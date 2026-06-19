import os
import re
import time
import json
import torch
import cv2
import numpy as np
import threading
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image
from collections import defaultdict

# PySide6 元件引入
from PySide6.QtCore import Signal, QObject

# Ultralytics 基礎模組引入
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.files import increment_path
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz, check_imshow, check_yaml
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.trackers import track

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    try:
        from ultralytics.utils.metrics import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

if not hasattr(ops, 'non_max_suppression'):
    ops.non_max_suppression = non_max_suppression

def seconds_to_hms(seconds):
    if seconds > 59:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
    else:
        hours = 0
        minutes = 0
    return hours, minutes, seconds


class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)       
    yolo2main_res_img = Signal(np.ndarray)       
    yolo2main_status_msg = Signal(str)           
    yolo2main_fps = Signal(str)                  
    yolo2main_labels = Signal(dict)              
    yolo2main_time = Signal(str)                 
    yolo2main_silder_range = Signal(str)         
    yolo2main_class_num = Signal(int)            
    yolo2main_target_num = Signal(int)           

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        self.done_warmup = False
        
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

        self.used_model_name = None  
        self.new_model_name = None   
        self.source = ''             
        self.stop_dtc = False        
        self.continue_dtc = True     
        self.save_res = False        
        self.save_txt = False        
        self.save_res_cam = False    
        self.save_txt_cam = False    
        self.iou_thres = 0.45        
        self.conf_thres = 0.25       
        self.speed_thres = 0         
        self.labels_dict = {}        
        self.progress_value = 0      
        self.task = ''
        self.stream_buffer = True

        self.model = None
        self.task_frist = ''
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self.frames = None
        self.frame = 0               
        self.fps = None
        
        self.start_time = None       
        self.elapsed_time = 0        
        
        self.model_initialized = False
        self._lock = threading.Lock() 
        
        callbacks.add_integration_callbacks(self)

    def initialize_model(self):
        with self._lock:  
            if not self.model_initialized:
                if self.task == 'Track':
                    self.track_model = YOLO(self.new_model_name)
                    self.model = self.track_model.predictor.model if hasattr(self.track_model.predictor, 'model') else self.track_model.model
                else:
                    self.setup_model(self.new_model_name)
                self.model_initialized = True
                self.used_model_name = self.new_model_name

    def check_model_update(self):
        if self.used_model_name != self.new_model_name:
            with self._lock:  
                if self.model_initialized:
                    if self.task == 'Track':
                        self.track_model = YOLO(self.new_model_name)
                        self.model = self.track_model.predictor.model if hasattr(self.track_model.predictor, 'model') else self.track_model.model
                    else:
                        self.setup_model(self.new_model_name)
                    self.used_model_name = self.new_model_name

    def reset_model(self):
        with self._lock:
            self.model_initialized = False
            self.setup_model(self.new_model_name)

    def run(self):
        try:
            if self.args.verbose:
                LOGGER.info('')

            self.yolo2main_status_msg.emit('模型載入中...')
            self.initialize_model()
            self.check_model_update()

            if self.task == 'Track':
                self.track_history = defaultdict(lambda: [])
            else:
                self.track_history = None

            is_folder = isinstance(self.source, list)
            if is_folder:
                for source in self.source:
                    self.setup_source(source)
                    self.set_video_total_time(source)
                    self.stream_inference()
            else:
                self.setup_source(self.source)
                self.set_video_total_time(self.source)
                self.stream_inference()

        except Exception as e:
            self.yolo2main_status_msg.emit(f'錯誤: {str(e)}')
            LOGGER.error(f'Error in run: {str(e)}')
            traceback.print_exc()

    @smart_inference_mode()
    def stream_inference(self, *args, **kwargs):
        try:
            self.check_save_dirs()
            
            if not self.done_warmup:
                is_pt = getattr(self.model, 'pytorch', False) or getattr(self.model, 'pt', False)
                is_triton = getattr(self.model, 'triton', False)
                self.model.warmup(imgsz=(1 if is_pt or is_triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.frames = self.dataset.frames if hasattr(self.dataset, 'frames') else None
            if isinstance(self.frames, list):
                self.frames = self.frames[0]

            self.seen, self.windows, self.batch = 0, [], None
            profilers = [ops.Profile(device=self.device) for _ in range(3)]
            
            while True:
                if 'obb' in self.used_model_name and self.task == 'Detect':
                    self.task = 'obb'
                    self.task_frist = 'Detect'
                if self.task == 'obb' and 'obb' not in self.used_model_name:
                    self.task = self.task_frist

                if self.continue_dtc:
                    if self.start_time is None:
                        self.start_time = datetime.now()
                    else:
                        self.elapsed_time += (datetime.now() - self.start_time).total_seconds()
                        self.start_time = datetime.now()

                    try:
                        batch = next(self.dataset)
                    except StopIteration:
                        break
                    
                    self.batch = batch
                    self.yolo2main_status_msg.emit('檢測中...')

                    paths, im0s, s = self.batch
                    im = self.preprocess_batch(im0s, profilers[0])
                    
                    with self._lock:  
                        if self.task == 'Track':
                            self.results = self.track_model.track(
                                source=im0s, conf=self.conf_thres, iou=self.iou_thres, 
                                persist=True, verbose=False
                            )
                            self.track_pointlist = []
                            for res in self.results:
                                if res.boxes.id is not None:
                                    boxes = res.boxes.xywh.cpu().numpy()
                                    track_ids = res.boxes.id.int().cpu().tolist()
                                    for box, track_id in zip(boxes, track_ids):
                                        x, y, w, h = box
                                        track_item = self.track_history[track_id]
                                        track_item.append((int(x), int(y)))
                                        if len(track_item) > 30:
                                            track_item.pop(0)
                                        points = np.array(track_item, dtype=np.int32).reshape((-1, 1, 2))
                                        self.track_pointlist.append(points)
                        else:
                            preds = self.inference(im, profilers[1], *args, **kwargs)
                            self.results = self.postprocess_results(preds, im, im0s, self.track_history, profilers[2])

                    self.run_callbacks('on_predict_postprocess_end')
                    self.handle_results(im, im0s, paths, s, profilers)

                else:
                    if self.start_time is not None:
                        self.elapsed_time += (datetime.now() - self.start_time).total_seconds()
                        self.start_time = None

                if self.check_completion():
                    break

                if self.stop_dtc:
                    self.release_video_writers()
                    self.yolo2main_status_msg.emit('檢測終止')
                    
                    # 安全檢查：只有當 dataset 具備 close 方法時才進行調用
                    if hasattr(self.dataset, 'close'):
                        self.dataset.close()
                        
                    break

            if not self.source_type.stream and (self.frames is None or self.frame is None):
                self.yolo2main_status_msg.emit('檢測完成')

            if self.start_time is not None:
                self.elapsed_time = 0
                self.start_time = None

        except Exception as e:
            self.yolo2main_status_msg.emit(f'錯誤: {str(e)}')
            LOGGER.error(f'Error in run: {str(e)}')
            traceback.print_exc()

    def preprocess_batch(self, im0s, profiler):
        with profiler:
            return self.classify_preprocess(im0s) if self.task == 'Classify' else self.preprocess(im0s)

    def postprocess_results(self, preds, im, im0s, track_history, profiler):
        with profiler:
            if self.task == 'Classify':
                return self.classify_postprocess(preds, im, im0s)
            else:
                postprocess_methods = {
                    'Detect': self.postprocess,
                    'obb': self.obb_postprocess,
                    'Segment': self.segment_postprocess,
                    'Pose': self.pose_postprocess,
                }
                return postprocess_methods[self.task](preds, im, im0s)

    def handle_results(self, im, im0s, paths, s, profilers):
        n = len(im0s)
        for i in range(n):
            self.seen += 1
            self.results[i].speed = {
                'preprocess': profilers[0].dt * 1E3 / n,
                'inference': profilers[1].dt * 1E3 / n,
                'postprocess': profilers[2].dt * 1E3 / n
            }
            self.class_nums = 0
            self.target_nums = 0
            
            s[i] += self.write_results(i, Path(paths[i]), im, s)
            im0 = None if self.source_type.tensor else im0s[i].copy()
            if 'no detections' in s:
                self.im = im0

            self.send_results(im0)
  
    def send_results(self, im0):
        """[GUI 傳輸最佳化補丁]：在發送訊號前進行合理的降採樣(Downsampling)，防範 4K 巨量資料阻塞主線程"""
        target_display_width = 1280  # 限制傳遞給 UI 的最大寬度，兼顧畫質與極致速度

        def scale_image(img_mat):
            if img_mat is not None and isinstance(img_mat, np.ndarray) and img_mat.size > 0:
                h, w = img_mat.shape[:2]
                if w > target_display_width:
                    scale = target_display_width / w
                    return cv2.resize(img_mat, (target_display_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                return img_mat.copy()
            return img_mat

        scaled_pre = scale_image(im0 if isinstance(im0, np.ndarray) else (im0[0] if im0 is not None else None))
        if scaled_pre is not None:
            self.yolo2main_pre_img.emit(scaled_pre)

        scaled_res = scale_image(self.im)
        if scaled_res is not None:
            self.yolo2main_res_img.emit(scaled_res)

        if self.task != 'Classify':
            self.yolo2main_class_num.emit(self.class_nums)
            self.yolo2main_target_num.emit(self.target_nums)
        if not isinstance(self.frames, list) and self.frames is not None:
            self.yolo2main_fps.emit(str(self.fps))
        if self.speed_thres != 0:
            time.sleep(self.speed_thres / 1000)
        self.set_video_current_time()

    def check_completion(self):
        if (self.frame == self.frames) and self.frames is not None and self.frame is not None:
            self.release_video_writers()
            self.yolo2main_status_msg.emit('檢測完成')
            return True
        elif self.source_type.stream and self.frames == self.frame + 1:
            self.yolo2main_status_msg.emit('檢測完成')
            return True
        return False

    def release_video_writers(self):
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

    def set_video_total_time(self, source):
        if isinstance(source, str) and source.endswith((".avi", ".mp4", ".mkv", ".mov")):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.duration = max(1, int(total_frames / fps)) if fps > 0 else 1
                self.total_hours, self.total_minutes, self.total_seconds = seconds_to_hms(self.duration)
                cap.release()  
        else:
            self.duration = 1
            self.total_hours, self.total_minutes, self.total_seconds = '', '', ''

    def set_video_current_time(self):
        if self.frames is not None and getattr(self.dataset, 'mode', '') != "stream" and self.dataset.mode != "stream":     
            new_fps = round(self.frames / self.duration, 2)
            current = round(self.frame / new_fps) if new_fps > 0 else 0
            current_hours, current_minutes, current_seconds = seconds_to_hms(current)
            current_time = f"{current_hours:02}:{current_minutes:02}:{current_seconds:02}"
            total_time = f"{self.total_hours:02}:{self.total_minutes:02}:{self.total_seconds:02}"
            self.yolo2main_time.emit(f"{current_time} / {total_time}")
            self.yolo2main_silder_range.emit(f"{current},{self.duration}")

        elif self.frames is None or getattr(self.dataset, 'mode', '') == "stream" or self.dataset.mode == "stream":
            self.cam_hours, self.cam_minutes, self.cam_seconds = seconds_to_hms(int(self.elapsed_time))
            cam_time = f"{self.cam_hours}:{self.cam_minutes:02}:{self.cam_seconds:02}"
            self.yolo2main_time.emit("程式運作時間：" + f"{cam_time}")
            self.yolo2main_silder_range.emit("0,1")

    def check_save_dirs(self):
        if self.save_res or self.save_txt or self.save_res_cam or self.save_txt_cam:
            (self.save_dir / 'labels' if (self.save_txt or self.save_txt_cam) else self.save_dir).mkdir(parents=True, exist_ok=True)

    def inference(self, im, profiler, *args, **kwargs):
        with profiler:
            visualize = (
                increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
                if self.args.visualize and (not self.source_type.tensor)
                else False
            )
            return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def preprocess(self, img):
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = torch.from_numpy(img).to(self.device, non_blocking=True)
            img = img.permute(0, 3, 1, 2)  
            img = img.half() if hasattr(self, 'model') and self.model.fp16 else img.float()
            img = img.flip(1) 
            img /= 255.0
        else:
            img = img.to(self.device, non_blocking=True)
            img = img.half() if self.model.fp16 else img.float()
        return img

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(
            preds, self.conf_thres, self.iou_thres,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det, classes=self.args.classes,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def obb_postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(
            preds, self.conf_thres, self.iou_thres,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            nc=len(self.model.names), classes=self.args.classes, rotated=True,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results

    def classify_preprocess(self, img):
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device, non_blocking=True)
        return img.half() if self.model.fp16 else img.float()

    def classify_postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def segment_postprocess(self, preds, img, orig_imgs):
        p = ops.non_max_suppression(
            preds[0], self.conf_thres, self.iou_thres,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            nc=len(self.model.names), classes=self.args.classes,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def pose_postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(
            preds, self.conf_thres, self.iou_thres,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            classes=self.args.classes, nc=len(self.model.names),
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts))
        return results

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        self.transforms = (
            getattr(self.model.model, "transforms", classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction))
            if self.task == "Classify" else None
        )
        self.dataset = load_inference_source(
            source=source, batch=self.args.batch, vid_stride=self.args.vid_stride, buffer=self.stream_buffer,
        )
        if self.dataset is not None and not hasattr(self.dataset, 'count'):
            self.dataset.count = 0

        self.source_type = self.dataset.source_type
        self.vid_writer = {}
        
    def write_results(self, i, p, im, s):
        string = ""
        if len(im.shape) == 3:
            im = im[None]

        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            if hasattr(self.dataset, 'count'):
                self.frame = self.dataset.count
            else:
                self.frame = getattr(self, 'seen', 0)
        else:
            match = re.search(r"frame (\d+)/", s[i])
            self.frame = int(match.group(1)) if match else None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{self.frame}"))
        string += f"%gx%g " % im.shape[2:]

        result = self.results[i]
        result.save_dir = str(self.save_dir)
        string += result.verbose() + f"{result.speed['inference']:.1f}ms"

        if self.task != 'Classify':
            det = result.boxes if self.task != 'obb' else result.obb
            if len(det) == 0:
                string += "(no detections)"
            else:
                for c in det.cls.unique():
                    n = (det.cls == c).sum()
                    self.target_nums += int(n)
                    self.class_nums += 1

        self.plotted_img = result.plot(
            line_width=self.args.line_width, boxes=self.args.show_boxes,
            conf=self.args.show_conf, labels=self.args.show_labels,
            im_gpu=None if self.args.retina_masks else im[i],
        )

        if self.save_txt or self.save_txt_cam:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))

        self.save_predicted_images(str(self.save_dir / p.name), self.frame)
        return string

    def save_predicted_images(self, save_path="", frame=0):
        self.im = self.plotted_img
        if self.task == 'Track' and hasattr(self, 'track_pointlist'):
            for points in self.track_pointlist:
                cv2.polylines(self.im, [points], isClosed=False, color=(203, 224, 252), thickness=4)

        if self.dataset.mode in {"stream", "video"}:
            self.fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.rsplit(".", 1)[0]}_frames/'

            if save_path not in self.vid_writer and (self.save_res or self.save_res_cam):
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=self.fps,
                    frameSize=(self.im.shape[1], self.im.shape[0]),
                )

            if self.save_res or self.save_res_cam:
                self.vid_writer[save_path].write(self.im)

            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", self.im)
        else:
            if self.save_res or self.save_res_cam:
                cv2.imwrite(save_path, self.im)
