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
"""Render video annotations onto a source video. Useful for analysis."""

from pathlib import Path

from absl import app
from absl import flags

import cv2
import sqlite3
import pandas as pd

from traffic_ml.lib.data      import COCO_CLASSES
from traffic_ml.lib.convertor import Annotations

from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages

FLAGS = flags.FLAGS
flags.DEFINE_string("source",      None,     "Video path")
flags.DEFINE_string("gt_annot",    None,     "Ground truth annotations")
flags.DEFINE_string("pred_annot",  None,     "(Optional) Predicted annotations for contrast")
flags.DEFINE_string("save_path",   "./out/", "Save path for annotated vid")
flags.DEFINE_integer("vid_stride", 1,        "(Optional) Plot strided annotations")
flags.mark_flag_as_required("source")
flags.mark_flag_as_required("gt_annot")

GREEN = [0, 255, 0]
RED   = [255, 0, 0]

def write_frame(r, im0, color):
    x, y, w, h, l, score, d_id = \
        int(r["bbox_x"]), \
        int(r["bbox_y"]), \
        int(r["bbox_w"]), \
        int(r["bbox_h"]), \
        r["label"], \
        r["conf"], \
        int(r["det_id"])
    cv2.rectangle(
        img=im0,
        pt1=(x, y),
        pt2=(x + w, y + h),
        color=color,
        thickness=2)
    cv2.putText(im0,
        f'{l}:{d_id:.3f}', (x, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75, [225, 255, 255],
        thickness=2)

def write_vid(src, dst, gt_annot, pred_annot):
    fps = 30

    images = LoadImages(src,
                        imgsz=640,
                        stride=32,
                        auto=True,
                        transforms=None,
                        vid_stride=1)

    gt_annots = Annotations.from_darwin(gt_annot)
    
    save_path = dst

    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (1920, 1080))
    
    if pred_annot:
        con = sqlite3.connect(pred_annot)
        detections_df = pd.read_sql_query("SELECT * FROM detection;", con)
        con.close()

    for f, img in enumerate(images):
        _, _, im0, _, _ = img

        annots = gt_annots[gt_annots["frame"] == f]

        for i, r in annots.iterrows():
            write_frame(r, im0, GREEN)
        
        if pred_annot and not f % FLAGS.vid_stride:
            pred_f = int(f / FLAGS.vid_stride)

            annots = detections_df[detections_df["frame"] == pred_f]
            for j, r in annots.iterrows():
                write_frame(r, im0, RED)
            
        writer.write(im0)

    writer.release()

def main(unused_argv):
    write_vid(FLAGS.source, FLAGS.save_path, FLAGS.gt_annot, FLAGS.pred_annot)

if __name__ == "__main__":
    app.run(main)