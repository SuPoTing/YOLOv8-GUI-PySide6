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

# Import PySide6 components
from PySide6.QtCore import Signal, QObject

# Import Ultralytics base modules
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, nms, ops
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
        self.im = None  # Ensure explicit declaration of property
        
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

            self.yolo2main_status_msg.emit('Loading model...')
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
            self.yolo2main_status_msg.emit(f'Error: {str(e)}')
            LOGGER.error(f'Error in run: {str(e)}')
            traceback.print_exc()

    @smart_inference_mode()
    def stream_inference(self, *args, **kwargs):
        try:
            self.check_save_dirs()
            
            if not self.done_warmup:
                # === Safe warmup patch: Automatically detect AutoBackend vs native nn.Module ===
                if hasattr(self.model, 'warmup'):
                    fmt = getattr(self.model, 'format', 'pt')
                    channels = getattr(self.model, 'channels', 3)
                    self.model.warmup(imgsz=(1 if fmt in {"pt", "triton"} else self.dataset.bs, channels, *self.imgsz))
                else:
                    # If it is a native DetectionModel extracted from a Track task, manually feed a dummy tensor to warmup the GPU
                    try:
                        dev = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device('cpu')
                        half_mode = getattr(self.model, 'fp16', False) or (hasattr(self.model, 'args') and getattr(self.model.args, 'half', False))
                        
                        dummy_input = torch.zeros(1, 3, *self.imgsz).to(dev)
                        if half_mode:
                            dummy_input = dummy_input.half()
                        
                        with torch.no_grad():
                            self.model(dummy_input)
                    except Exception as warmup_err:
                        LOGGER.warning(f"Background model manual warmup skipped or failed (does not affect subsequent detection): {warmup_err}")
                
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
                    self.yolo2main_status_msg.emit('Detecting...')

                    paths, im0s, s = self.batch
                    im = self.preprocess_batch(im0s, profilers[0])
                    
                    with self._lock:  
                        if self.task == 'Track':
                            # === Fix patch: Enable profiler for Track tasks to prevent crash from missing '.dt' in handle_results ===
                            with profilers[1]:
                                self.results = self.track_model.track(
                                    source=im0s, conf=self.conf_thres, iou=self.iou_thres, 
                                    persist=True, verbose=False
                                )
                            
                            # Manually assign safe default values to profilers[0] and profilers[2] to prevent errors in handle_results
                            if not hasattr(profilers[0], 'dt'): profilers[0].dt = 0.0
                            if not hasattr(profilers[2], 'dt'): profilers[2].dt = 0.0
                            
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
                            # Regular detection and segmentation pipeline
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
                    self.yolo2main_status_msg.emit('Detection terminated')
                    
                    if hasattr(self.dataset, 'close'):
                        self.dataset.close()
                        
                    break

            if not self.source_type.stream and (self.frames is None or self.frame is None):
                self.yolo2main_status_msg.emit('Detection completed')

            if self.start_time is not None:
                self.elapsed_time = 0
                self.start_time = None

        except Exception as e:
            self.yolo2main_status_msg.emit(f'Error: {str(e)}')
            LOGGER.error(f'Error in run: {str(e)}')
            traceback.print_exc()

    def preprocess_batch(self, im0s, profiler):
        with profiler:
            return self.classify_preprocess(im0s) if self.task == 'Classify' else self.preprocess(im0s)

    def postprocess_results(self, preds, im, im0s, track_history, profiler):
        with profiler:
            if self.task == 'Classify':
                return self.classify_postprocess(preds, im, im0s)
            elif self.task == 'Segment':
                # Force Segment task to use its exclusive postprocess pipeline to correctly separate protos
                return self.segment_postprocess(preds, im, im0s)
            else:
                return self.postprocess(preds, im, im0s)

    def handle_results(self, im, im0s, paths, s, profilers):
        # Guard against None or empty self.results
        if self.results is None or len(self.results) == 0:
            return

        # === Fix patch: Use a safe range length to prevent IndexError in Track tasks ===
        n = len(im0s)
        safe_n = min(n, len(self.results))
        
        for i in range(safe_n):
            self.seen += 1
            
            # Safely assign speed dictionary
            if hasattr(self.results[i], 'speed') or isinstance(self.results[i], Results):
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n
                }
                
            self.class_nums = 0
            self.target_nums = 0
            
            with self._lock:
                s[i] += self.write_results(i, Path(paths[i]), im, s)
                im0 = None if self.source_type.tensor else im0s[i].copy()
                if 'no detections' in s:
                    self.im = im0

            self.send_results(im0)
  
    def send_results(self, im0):
        """[GUI Transmission Optimization Patch]: Introduces thread lock and deep copy 
        to prevent pointer conflicts caused by delayed reads from the main thread during scaling."""
        target_display_width = 1280  
        raw_pre, raw_res = None, None

        with self._lock:
            if im0 is not None:
                raw_pre = im0 if isinstance(im0, np.ndarray) else (im0[0] if len(im0) > 0 else None)
            if self.im is not None:
                raw_res = self.im

        def scale_and_copy_image(img_mat):
            if img_mat is not None and isinstance(img_mat, np.ndarray) and img_mat.size > 0:
                h, w = img_mat.shape[:2]
                if w > target_display_width:
                    scale = target_display_width / w
                    resized = cv2.resize(img_mat, (target_display_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    return resized.copy()  
                return img_mat.copy()      
            return None

        scaled_pre = scale_and_copy_image(raw_pre)
        scaled_res = scale_and_copy_image(raw_res)

        if scaled_pre is not None:
            self.yolo2main_pre_img.emit(scaled_pre)

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
            self.yolo2main_status_msg.emit('Detection completed')
            return True
        elif self.source_type.stream and self.frames == self.frame + 1:
            self.yolo2main_status_msg.emit('Detection completed')
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
            self.yolo2main_time.emit("Program running time: " + f"{cam_time}")
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
        
        # === Safe precision reading patch: Prevent crashes when DetectionModel lacks fp16 attribute ===
        is_model_ready = hasattr(self, 'model') and self.model is not None
        if is_model_ready:
            if hasattr(self.model, 'fp16'):
                # Directly read attribute if it is an officially wrapped AutoBackend
                half_mode = self.model.fp16
            else:
                # Check weight tensor type to determine if it's half-precision for native DetectionModel from Track tasks
                try:
                    half_mode = next(self.model.parameters()).dtype == torch.float16 if list(self.model.parameters()) else False
                except Exception:
                    half_mode = getattr(self.args, 'half', False)
        else:
            half_mode = getattr(self.args, 'half', False)

        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = torch.from_numpy(img).to(self.device, non_blocking=True)
            img = img.permute(0, 3, 1, 2)  
            img = img.half() if half_mode else img.float()
            img = img.flip(1) 
            img /= 255.0
        else:
            img = img.to(self.device, non_blocking=True)
            img = img.half() if half_mode else img.float()
        return img

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        save_feats = getattr(self, "_feats", None) is not None
        
        # If tuple, the first element is usually the core prediction Tensor for object detection
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = nms.non_max_suppression(
            preds,
            self.conf_thres,
            kwargs.pop("iou", self.iou_thres),  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=True if self.task == "obb" else False,
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]
            
        if self.task == "Segment":
            protos = kwargs.pop("protos", None)
            results = self.segment_construct_results(preds, img, orig_imgs, protos=protos)
        else:
            results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        if self.task == "Detect":
            return [
                self.construct_result(pred, img, orig_img, img_path)
                for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
            ]
        elif self.task == "Pose":
            return [
                self.pose_construct_result(pred, img, orig_img, img_path)
                for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
            ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

    def obb_construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and rotation angles with shape (N, 7) where the
                last dimension contains [x, y, w, h, confidence, class_id, angle].
            img (torch.Tensor): The image after preprocessing with shape (B, C, H, W).
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and oriented bounding
                boxes.
        """
        rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)

    def classify_preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # Convert uint8 to fp16/32

    def classify_postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):  # Input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def segment_postprocess(self, preds, img, orig_imgs):
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return self.postprocess(preds[0], img, orig_imgs, protos=protos)

    def segment_construct_results(self, preds, img, orig_imgs, protos):
        """Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (torch.Tensor): Prototype masks tensor with shape (B, C, H, W).

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names, bounding
                boxes, and masks.
        """
        return [
            self.segment_construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def segment_construct_result(self, pred, img, orig_img, img_path, proto):
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not all(keep):  # most predictions have masks
                pred, masks = pred[keep], masks[keep]  # indexing is slow
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    def pose_construct_result(self, pred, img, orig_img, img_path):
        result = self.construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        
        # Remove crop_fraction parameter; newer versions of classify_transforms do not accept it
        self.transforms = (
            getattr(self.model.model, "transforms", classify_transforms(self.imgsz[0]))
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
