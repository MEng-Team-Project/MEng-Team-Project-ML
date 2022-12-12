# MIT License
# 
# Copyright (c) 2022 MEng-Team-Project
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

import time
import os
import subprocess

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string ("video_dir", None, "Directory of video files to bulk analyse")
flags.DEFINE_boolean("half",      True, "Run yolo in half precision")
flags.DEFINE_boolean("tracking",  True, "Track objects across frames in videos as well")
flags.DEFINE_integer("size",      640,  "Yolo image input size")
flags.mark_flag_as_required("video_dir")

OFFLINE_ANALYSIS = lambda source, half, size, tracking: \
    f'python yolov7-segmentation/segment/predict.py --weights ./yolov7-seg.pt --source {source} --{half} --save-txt --save-conf --img-size {size} {tracking}'

def main(unused_argv):
    video_dir = FLAGS.video_dir
    videos    = os.listdir(video_dir)

    for i, video in enumerate(videos):
        print(video)
        
        source    = os.path.join(video_dir, video)
        half      = FLAGS.half
        half      = "half" if half else ""
        tracking  = FLAGS.tracking
        tracking  = "--trk" if tracking else ""
        size      = FLAGS.size

        args = list(filter(None, OFFLINE_ANALYSIS(source, half, size, tracking).split(" ")))

        print("args:", args)

        start_time = time.time()

        # Pipeing the output will run this command async
        subprocess.run(
            args,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            cwd=os.getcwd())

        end_time = time.time() - start_time

        print(i+1, video, end_time)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)