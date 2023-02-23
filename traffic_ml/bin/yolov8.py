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
the object detections and ClassySORT object tracking information."""

import os

import numpy as np

from absl import app
from absl import flags

from ultralytics import YOLO
from pytorch_lightning.callbacks import Callback
from ultralytics.yolo import utils
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.configs import get_config

from traffic_ml.lib.sort_count import Sort

FLAGS = flags.FLAGS
flags.DEFINE_string("video_dir", None, "Directory of video files to bulk analyse")
flags.DEFINE_string("model", "yolov8n.pt", "Choose YOLOv8 model")

class TestCallback(Callback):
    def on_predict_start(self):
        print("ello")

def main(unused_argv):
    video_dir = FLAGS.video_dir
    videos    = os.listdir(video_dir)
    videos    = [video for video in videos
                 if video.endswith(".mp4")]
    # print(video_dir, videos)

    # Load the model
    cfg = get_config(DEFAULT_CONFIG)
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    predictor = DetectionPredictor(cfg)
    
    # Initialise SORT
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)

    # Print log output
    log = True

    # model = YOLO("yolov8n.pt")
    for i, video in enumerate(videos):
        print(video)
        
        # Use the model
        source  = os.path.join(video_dir, video)
        results = predictor.__call__(
            source,
            model=FLAGS.model,
            return_outputs=True,
            log=False) # batch

        for frame_idx, result in enumerate(results):
            frame_idx += 1
            
            # format per det (x1, y1, x2, y2, conf, cls)
            # print(result)
            dets = result["det"]
            print("det count:", len(dets))
            print(dets[0])
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
                

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)