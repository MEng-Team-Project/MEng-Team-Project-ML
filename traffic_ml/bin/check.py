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
"""Performs basic sanity test on draft format for text output for traffic
video analytics."""

import json
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot

from absl import app
from absl import flags    

FLAGS = flags.FLAGS
flags.DEFINE_string ("analysis_dir", None, "Directory which yolo model is storing analysis outputs")
flags.DEFINE_string ("fname", None, "Video filename with extension")
flags.mark_flag_as_required("analysis_dir")
flags.mark_flag_as_required("fname")

def main(unused_argv):
    # base_dir   = "C:\\Users\\win8t\\OneDrive\\Desktop\\projects\\traffic-core\\yolov7-segmentation\\runs\\predict-seg\\exp2\\labels\\"
    # fname      = "SEM_ID_TRAFFIC_TEST_TILTON_TINY"
    base_dir   = FLAGS.analysis_dir
    fname      = FLAGS.fname
    info_path  = lambda idx: f"{fname}_info_{idx}.json"
    base_path  = lambda idx: f"{fname}_{idx}.json"
    track_path = lambda idx: f"{fname}_track_{idx}.json"

    a_s = [], b_s = [], c_s = []

    for i in range(1, 229+1):
        idx = i
        with open(os.path.join(base_dir, info_path(idx))) as f:
            content = f.readlines()
            obj = json.loads(content[0])
            a = len(obj["infos"])
            a_s.append(a)
            #print("info:", len(obj["infos"]))
            #print([info for info in obj["infos"]])
        with open(os.path.join(base_dir, base_path(idx))) as f:
            content = "[" + ",".join(f.readlines()) + "]"
            obj = json.loads(content)
            b = len(obj)
            b_s.append(b)
            #print("base:", len(obj))
            #print([info for info in obj])
        with open(os.path.join(base_dir, track_path(idx))) as f:
            content = f.readlines()
            obj = json.loads(content[0])
            c = len(obj["routes"])
            c_s.append(c)
            # print("routes:", len(obj["routes"]))
        print(i, a, b, c)

    ax = subplot(1,1,1)
    ax.plot(a_s, label="Tracked Detections")
    ax.plot(b_s, label="Segments")
    ax.plot(c_s, label="Routes")
    plt.legend()
    plt.show()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)