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
"""Bulk scrape an entire directory of traffic video footage to identify vehicles,
count each category of vehicle and to track the routes of those vehicles
through traffic routes / junctions."""

import os
import time
from absl import app
from absl import flags

from ultralytics import YOLO
from pytorch_lightning.callbacks import Callback
from ultralytics.yolo import utils
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.configs import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string("video_dir", None, "Directory of video files to bulk analyse")

class TestCallback(Callback):
    def on_predict_start(self):
        print("ello")

def main(unused_argv):
    video_dir = FLAGS.video_dir
    videos    = os.listdir(video_dir)
    videos    = [video for video in videos
                 if video.endswith(".mp4")]
    print(video_dir, videos)

    # Load the model
    cfg = get_config(DEFAULT_CONFIG)
    # cfg.source = video_dir
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    predictor = DetectionPredictor(cfg)
    """
    predictor.setup(
        source=os.path.join(video_dir, videos[0]),
        model="yolov8n.pt") # yolov8x.pt
    """

    # predictor.predict_cli()
    
    # model = YOLO("yolov8n.pt")
    for i, video in enumerate(videos):
        print(video)
        
        # Use the model
        source  = os.path.join(video_dir, video)
        results = predictor.__call__(
            source,
            model="yolov8n.pt",
            return_outputs=True) # batch

        for frame_idx, result in enumerate(results):
            # format per det (x1, y1, x2, y2, conf, cls)
            # print(result)
            dets = result["det"]
            for det in dets:
                print(frame_idx, det)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)