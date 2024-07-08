####################ultralytics==8.2.22####################
####################cuda11.8####################
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
from ultralytics import YOLO

from PySide6.QtCore import Signal, QObject
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import threading
import traceback
import re
import time
import json
import torch
import cv2
import os

class YoloPredictor(BasePredictor, QObject):
    # Signal definitions for communication with other parts
    yolo2main_pre_img = Signal(np.ndarray)   # Original image signal
    yolo2main_res_img = Signal(np.ndarray)   # Test result signal
    yolo2main_status_msg = Signal(str)       # Detection/pause/stop/test completed/error report signal
    yolo2main_fps = Signal(str)              # Frame rate signal
    yolo2main_labels = Signal(dict)          # Detection target results (number of each class)
    yolo2main_progress = Signal(int)         # Progress signal
    yolo2main_class_num = Signal(int)        # Number of detected classes
    yolo2main_target_num = Signal(int)       # Number of detected targets

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # Call the initialization method of the parent class
        super(YoloPredictor, self).__init__()
        # Initialize PyQt's QObject
        QObject.__init__(self)

        # Parse the configuration file
        self.args = get_cfg(cfg, overrides)
        # Set the model save directory
        self.save_dir = get_save_dir(self.args)
        # Initialize a flag to mark whether the model has completed warm-up
        self.done_warmup = False
        # Check if images should be displayed
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

        # GUI-related attributes
        self.used_model_name = None  # Name of the detection model to be used
        self.new_model_name = None   # Name of the model changed in real-time
        self.source = ''             # Input source
        self.stop_dtc = False        # Flag to terminate detection
        self.continue_dtc = True     # Flag to pause detection
        self.save_res = False        # Flag to save test results
        self.save_txt = False        # Flag to save label (txt) files
        self.save_res_cam = False    # Flag to save webcam test results
        self.save_txt_cam = False    # Flag to save webcam label (txt) files
        self.iou_thres = 0.45        # IoU threshold
        self.conf_thres = 0.25       # Confidence threshold
        self.speed_thres = 0         # Delay, milliseconds
        self.labels_dict = {}        # Dictionary returning detection results
        self.progress_value = 0      # Value of the progress bar
        self.task = ''

        # Properties that can be used once setup is completed
        self.model = None
        self.data = self.args.data  # data_dict
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
        self.fps = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    # main for detect
    @smart_inference_mode()
    def run(self, *args, **kwargs):
        try:
            if self.args.verbose:
                LOGGER.info('')
            # Setup model
            self.yolo2main_status_msg.emit('Loading model...')
            if not self.model:
                if self.task == 'Track':
                    track_model = YOLO(self.new_model_name)
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name
            with self._lock:  # for thread-safe inference
                if self.task == 'Track':
                    track_history = defaultdict(lambda: [])
                # Setup source every time predict is called
                self.setup_source(self.source if self.source is not None else self.args.source)

                # Check save path/labels
                if self.save_res or self.save_txt or self.save_res_cam or self.save_txt_cam:
                    (self.save_dir / 'labels' if (self.save_txt or self.save_txt_cam) else self.save_dir).mkdir(parents=True, exist_ok=True)

                # Model warm-up
                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                    self.done_warmup = True

                self.seen, self.windows, self.batch = 0, [], None
                profilers = (
                    ops.Profile(device=self.device),
                    ops.Profile(device=self.device),
                    ops.Profile(device=self.device),
                )

                batch = iter(self.dataset)
                while True:
                    # for self.batch in self.dataset: 
                    # Change model midway
                    if self.used_model_name != self.new_model_name:  
                        # self.yolo2main_status_msg.emit('Change Model...')
                        if self.task == 'Track':
                            track_model = YOLO(self.used_model_name)
                        self.setup_model(self.new_model_name)
                        self.used_model_name = self.new_model_name

                    if 'obb' in self.used_model_name and self.task == 'Detect':
                        self.task = 'obb'
                        self.task_frist = 'Detect'
                    if self.task == 'obb' and 'obb' not in self.used_model_name:
                        self.task = self.task_frist

                    # Pause switch
                    if self.continue_dtc:
                        try:
                            batch = next(self.dataset)  # next data
                        except StopIteration:
                            break
                        self.batch = batch
                        self.yolo2main_status_msg.emit('Detecting...')

                        paths, im0s, s = self.batch

                        # Preprocess
                        with profilers[0]:
                            if self.task == 'Classify':
                                im = self.classify_preprocess(im0s)
                            else:
                                im = self.preprocess(im0s)                                
                        # Inference
                        with profilers[1]:
                            preds = self.inference(im, *args, **kwargs)
                        # Postprocess
                        with profilers[2]:
                            if self.task == 'Classify':
                                self.results = self.classify_postprocess(preds, im, im0s)
                            elif self.task == 'Detect':
                                self.results = self.postprocess(preds, im, im0s)
                            elif self.task == 'obb':
                                self.results = self.obb_postprocess(preds, im, im0s)
                            elif self.task == 'Segment':
                                self.results = self.segment_postprocess(preds, im, im0s)
                            elif self.task == 'Pose':
                                self.results = self.pose_postprocess(preds, im, im0s)
                            elif self.task == 'Track':
                                self.results, self.track_pointlist = self.track_postprocess(track_model, track_history, preds, im, im0s)
                        self.run_callbacks('on_predict_postprocess_end')
                        # Visualize, save, write results
                        n = len(im0s)
                        for i in range(n):
                            self.seen += 1
                            self.results[i].speed = {
                                'preprocess': profilers[0].dt * 1E3 / n,
                                'inference': profilers[1].dt * 1E3 / n,
                                'postprocess': profilers[2].dt * 1E3 / n}

                            self.class_nums = 0
                            self.target_nums = 0
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                            im0 = None if self.source_type.tensor else im0s[i].copy()
                            if 'no detections' in s:
                                self.im = im0

                            if isinstance(self.frame, int) and (not isinstance(self.frames, list) and self.frames is not None):
                                self.progress_value = int(self.frame/self.frames*1000)
                            elif not self.source or self.frames is None or self.frame is None:
                                # self.frame = 0
                                # self.frames = 1
                                self.progress_value = int(1000)
                            # Send test results
                            self.yolo2main_pre_img.emit(im0 if isinstance(im0, np.ndarray) else im0[0])   # Before detection
                            self.yolo2main_res_img.emit(self.im) # After detection
                            if self.task != 'Classify':
                                self.yolo2main_class_num.emit(self.class_nums)
                                self.yolo2main_target_num.emit(self.target_nums)
                            if not isinstance(self.frames, list) and self.frames is not None:
                                self.yolo2main_fps.emit(str(self.fps))
                            if self.speed_thres != 0:
                                time.sleep(self.speed_thres/1000)   # Delay, milliseconds
                        self.yolo2main_progress.emit(self.progress_value)   # Progress bar
                    if (self.frame == self.frames) and self.frames is not None and self.frame is not None:
                        for v in self.vid_writer.values():
                            if isinstance(v, cv2.VideoWriter):
                                v.release()
                        self.yolo2main_status_msg.emit('Detection completed!')
                        break
                    # Terminate detection flag detection
                    if self.stop_dtc:
                        for v in self.vid_writer.values():
                            if isinstance(v, cv2.VideoWriter):
                                v.release()
                        self.yolo2main_status_msg.emit('Detection terminated!')
                        break
                    # Print final results
                    if self.args.verbose and self.seen:
                        t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image

                    if self.save_txt or self.save_txt_cam or self.args.save_crop:
                        nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
                        s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if (self.save_txt or self.save_txt_cam) else ""
                if not self.source_type.stream and self.frames is None or self.frame is None:
                    self.yolo2main_status_msg.emit('Detection completed!')
        except Exception as e:
            pass
            traceback.print_exc()
            print(f"Error: {e}")
            self.yolo2main_status_msg.emit('%s' % e)

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
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
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def obb_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            rotated=True,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            # xywh, r, conf, cls
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results
        
    def classify_preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
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

    def segment_postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
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

    def pose_postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

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
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results

    def track_postprocess(self, model, track_history, preds, img, orig_imgs):
        # Set the track model for track line
        track_result = model.track(orig_imgs, persist=True)

        # Set the track preds
        preds = ops.non_max_suppression(preds,
                                         self.conf_thres,
                                         self.iou_thres,
                                         agnostic=self.args.agnostic_nms,
                                         max_det=self.args.max_det,
                                         classes=self.args.classes,
                                         nc=len(self.model.names))

        if not isinstance(orig_imgs,
                          list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            # Store result
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=track_result[0].boxes.data))
            
            # Get the boxes and track IDs
            boxes = track_result[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = track_result[0].boxes.id.int().cpu().tolist()
            output = []
            # Plot the tracks
            if results[0].boxes.id is not None:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                # Get the points
                    points = np.hstack(track).astype(np.int32).reshape(
                    (-1, 1, 2))
                    output.append(points)
        return results, output

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction)
            )
            if self.task == "Classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )

        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            self.frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            self.frame = int(match.group(1)) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{self.frame}"))
        string += "%gx%g " % im.shape[2:]
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += result.verbose() + f"{result.speed['inference']:.1f}ms"

        if self.task == 'Classify':
            prob = result.probs
            # for c in prob.top5:
            #     print(c)
        elif self.task not in ['Classify', 'obb']:
            det = result.boxes
            if len(det) == 0:
                string += f"(no detections)"

            for c in det.cls.unique():
                n = (det.cls == c).sum()  # detections per class
                self.target_nums += int(n)
                self.class_nums += 1
        elif self.task == 'obb':
            det = result.obb
            if len(det) == 0:
                string += f"(no detections)"

            for c in det.cls.unique():
                n = (det.cls == c).sum()  # detections per class
                self.target_nums += int(n)
                self.class_nums += 1

        # Add predictions to image
        self.plotted_img = result.plot(
            line_width=self.args.line_width,
            boxes=self.args.show_boxes,
            conf=self.args.show_conf,
            labels=self.args.show_labels,
            im_gpu=None if self.args.retina_masks else im[i],
        )

        # Save results
        if self.save_txt or self.save_txt_cam:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        
        self.save_predicted_images(str(self.save_dir / p.name), self.frame)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        """Save video predictions as mp4 at specified path."""
        # Save the image to be saved
        self.im = self.plotted_img
        # If the task is tracking, draw the tracking points
        if self.task == 'Track':
            for points in self.track_pointlist:
                cv2.polylines(self.im, [points],
                              isClosed=False,
                              color=(203, 224, 252),
                              thickness=10)
        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            self.fps = self.dataset.fps if self.dataset.mode == "video" else 30
            self.frames = self.dataset.frames
            # Create a directory to store frames if it doesn't exist
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if save_path not in self.vid_writer and (self.save_res or self.save_res_cam):  # New video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                # Initialize a VideoWriter for the new video
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=self.fps,
                    frameSize=(self.im.shape[1], self.im.shape[0]),  # (width, height)
                )
            # Write the frame to the video if it should be saved
            if self.save_res or self.save_res_cam:
                self.vid_writer[save_path].write(self.im)
            # Save the frame as an image if requested
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{self.frame}.jpg", self.im)

        # Save images
        if self.save_res or self.save_res_cam:
            cv2.imwrite(save_path, self.im)
