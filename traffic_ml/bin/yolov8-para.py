# MIT License
# 
# Copyright (c) 2023 MEng-Team-Project
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Bulk scrape an entire directory for its mp4 video files and extract
the object detections and ClassySORT object tracking information. Then
perform object detection in parallel by batching the input images
and then tracking using the ClassySort Kalman Filter on batches
of detections at a time."""

import os
import time
import glob

import numpy as np

from absl import app
from absl import flags

from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics import YOLO
from pytorch_lightning.callbacks import Callback
from ultralytics.yolo import utils
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils import LOGGER, colorstr, is_colab, is_kaggle, ops
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.checks import check_requirements
import math
import cv2
from threading import Thread
from urllib.parse import urlparse

from dotenv import dotenv_values
import neptune.new as neptune

from pathlib import Path
# from traffic_ml.lib.sort_count import Sort
from sort_count import Sort

import torch
torch.backends.cudnn.benchmark = False

from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow

FLAGS = flags.FLAGS
flags.DEFINE_string("video_dir", None, "Directory of video files to bulk analyse")
flags.DEFINE_string("model", "yolov8n.pt", "Pre-trained YoloV8 model")
flags.DEFINE_string("exp", "", "(Optional) Experiment name")
flags.DEFINE_integer("batch_size", 1, "Batch size during inference")
flags.DEFINE_integer("imgsz", 640, "NxN image size reshaping for Yolov8 object detection")
flags.DEFINE_bool("trk", False, "Whether or not to enable object tracking")
flags.mark_flag_as_required("video_dir")

class ParaLoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        self.imgs = []

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            # print("__next im0 shape:", im0.shape)
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        """
        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s
        """

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([LetterBox(self.imgsz, self.auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        # print("cv2.CAP_PROP_FRAME_COUNT, self.vid_stride, self.frames:", self.cap.get(cv2.CAP_PROP_FRAME_COUNT), self.vid_stride, self.frames)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files

def create_batch(dataset, batch_size=1):
    # self.sources, im, im0, None, ''
    sources_s = []
    im0_s = []
    im_s = []
    for _ in range(batch_size):
        try:
            sources, im, im0, _, _ = next(dataset)
            for src in sources:
                sources_s.append(src)
            im_s.append(im.copy())
            im0_s.append(im0.copy())
        except StopIteration:
            im_s.append(np.zeros_like(im_s[0].copy()))
            im0_s.append(np.zeros_like(im0_s[0].copy()))

    im_s  = np.stack([im for im in im_s])
    im0_s = np.stack([im0 for im0 in im0_s])
    return sources, im_s, im0_s, None, ""

# Batched inputs
class ParaDetectionPredictor(DetectionPredictor):
    def __init__(self, config=DEFAULT_CONFIG, overrides=None, batch_size=1, run=None):
        super().__init__(config, overrides)
        self.batch_size = batch_size

        # BENCHMARKING
        self.total_infer = 0
        self.run = run

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape # if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds): # , batch):
        self.seen += 1
        
        det = preds[idx]
        if self.return_outputs:
            if "det" not in self.output:
                self.output["det"] = []
            self.output["det"].append(det.cpu().numpy())

        return "" # log_string

    def setup(self, source=None, model=None, return_outputs=True):
        print("MY SETUP")
        # source
        source = str(source if source is not None else self.args.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # model
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
        stride, pt = model.stride, model.pt
        imgsz = check_imgsz(self.args.imgsz, stride=stride)  # check image size

        # Dataloader
        self.dataset = LoadImages(source,
                                    imgsz=imgsz,
                                    stride=stride,
                                    auto=pt,
                                    transforms=getattr(model.model, 'transforms', None),
                                    vid_stride=1)
        bs = len(self.dataset)
        self.vid_path, self.vid_writer = [None] * 1, [None] * 1
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

        self.model = model
        self.webcam = webcam
        self.screenshot = screenshot
        self.imgsz = imgsz
        self.done_setup = True
        self.device = device
        self.return_outputs = return_outputs

        return model

    @smart_inference_mode()
    def __call__(self, source=None, model=None, return_outputs=True, log=True):
        run = self.run
        print("My custom __call__")
        self.run_callbacks("on_predict_start")
        model = self.model if self.done_setup else self.setup(source, model, return_outputs)
        model.eval()
        self.seen, self.windows, self.dt = 0, [], (ops.Profile(), ops.Profile(), ops.Profile())
        first = True
        #print("dataset.shape", len(self.dataset))
        #for i, batch in enumerate(self.dataset):
        batch_size = self.batch_size
        batch_idxs = int(math.ceil(self.dataset.frames / batch_size))
        #print("batch_idxs:", batch_idxs)
        self.dataset.__iter__()
        for batch_idx in range(batch_idxs):
            batch = create_batch(self.dataset, batch_size=batch_size)
            # print(f"BATCH IDX: {batch_idx}")
            if first:
                pass
                # print("batch.shape:", len(batch), batch[1].shape, batch[2].shape)
                # first = False

            self.run_callbacks("on_predict_batch_start")
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                preds = model(im, augment=self.args.augment, visualize=visualize)
                # print("FIRST PRED B4:", preds)

            # postprocess
            with self.dt[2]:
                preds = self.postprocess(preds, im, im0s)

            for i in range(len(im)):
                p = Path(path)
                s += self.write_results(i, preds)

            if self.return_outputs:
                # print("len(self.output):", len(self.output))
                yield self.output
                self.output.clear()

            if log:
                self.total_infer += self.dt[1].dt * 1E3
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(preds) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

            self.run_callbacks("on_predict_batch_end")

        # Print results
        
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        self.t = list(t)
        run["mean_pre_process_ms"]   = self.t[0]
        run["mean_infer_ms"]         = self.t[1]
        run["mean_post_process_ms"]  = self.t[2]
        run["total_infer_ms"]        = self.total_infer

        if log:
            LOGGER.info(
                f'Speed: %.1fms pre-process, %.1fms inference, %.1fms postprocess per image at shape {(1, 3, *self.imgsz)}'
                % t)
            LOGGER.info(
                f'Speed: %.1fms total infer (only) {(self.total_infer)}')
        if self.args.save_txt or self.args.save:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")

def main(unused_argv):
    config  = dotenv_values(".env")
    project = config["NEPTUNE_NAME"]
    key     = config["NEPTUNE_KEY"]

    run = neptune.init_run(
        project=project,
        api_token=key)

    track_total = []
    track_tm = ops.Profile()

    video_dir = FLAGS.video_dir
    videos    = os.listdir(video_dir)
    videos    = [video for video in videos
                 if video.endswith(".mp4")]
    # print(video_dir, videos)

    # DEBUG
    log = False
    trk = FLAGS.trk
    batch_size = FLAGS.batch_size
    model = FLAGS.model
    imgsz = FLAGS.imgsz
    exp = FLAGS.exp
    run["model"] = model
    run["batch_size"] = batch_size
    run["trk"] = trk
    run["imgsz"] = imgsz
    run["exp"] = exp
    run["name"] = f"bs_{batch_size}_trk_{trk}_imgsz_{imgsz}"
    cfg = get_config(DEFAULT_CONFIG)
    cfg.imgsz = check_imgsz(imgsz, min_dim=2)
    predictor = ParaDetectionPredictor(cfg, batch_size=batch_size, run=run)
    
    # Initialise SORT
    sort_max_age    = 5
    sort_min_hits   = 2
    sort_iou_thresh = 0.2
    sort_tracker    = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    
    # model = YOLO("yolov8n.pt")
    for i, video in enumerate(videos):
        print(video)
        
        # Use the model
        source  = os.path.join(video_dir, video)
        results = predictor.__call__(
            source,
            model=model,
            return_outputs=True,
            log=True) # batch

        for frame_idx, result in enumerate(results):
            frame_idx += 1

            dets_s = result["det"]
            if trk:
                for dets in dets_s:
                    with track_tm:
                        dets_to_sort = np.empty((0,6))
                        if len(dets.shape) == 1:
                            dets = np.expand_dims(dets, axis=0)
                        for x1,y1,x2,y2,conf,detclass in dets[:, :6]:
                            dets_to_sort = np.vstack((dets_to_sort, 
                                            np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                        tracked_dets = sort_tracker.update(dets_to_sort)
                        tracks       = sort_tracker.getTrackers()

                        if log and frame_idx < 3: # 2nd frame
                            print("tracked_dets:", tracked_dets)
                            print("tracks:")

                        for track in tracks:
                            centroids = track
                            track_out = [[
                                centroids.centroids[i][0],
                                centroids.centroids[i][1],
                                centroids.centroids[i+1][0],
                                centroids.centroids[i+1][1]
                            ]
                            for i, _ in  enumerate(centroids.centroids)
                            if i < len(centroids.centroids)-1]
                            if log and frame_idx < 3: # 2nd frame
                                print("track->centroids:", track_out)

                        for i, det in enumerate(dets):
                            if log and frame_idx < 3: # 2nd frame
                                print("det", frame_idx, det)
                    track_total.append(track_tm.dt * 1E3)

    run["mean_track_ms"] = sum(track_total) / len(track_total)
    run["total_track_ms"] = sum(track_total)

    run.stop()
    
def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)