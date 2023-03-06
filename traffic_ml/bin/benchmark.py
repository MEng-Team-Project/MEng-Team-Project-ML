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
"""Benchmarks ML pipeline deployments across a range of hyperparameters."""

import os
import json
import subprocess

import itertools

from pathlib import Path

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("source",    None,     "Video path")
flags.DEFINE_string("save_dir",  "./out/", "Save directory for annotated vid")
flags.DEFINE_string("test_path", None,
                    "Path to benchmarking test conditions\n"
                    "Refer to README.md#Benchmarking for more details")

flags.mark_flag_as_required("source")
flags.mark_flag_as_required("test_path")

def sanitise_fname(fname):
    return str(Path(fname).stem).replace(".", "-")

def run_analysis(source, imgsz, vid_stride, model, tracker, analysis_path, half=True):
    args = [
        "python", "yolov8_tracking/track.py",
        "--source", source,
        "--yolo-weights", model,
        "--imgsz", str(imgsz),
        "--tracking-method", tracker,
        "--vid-stride", str(vid_stride),
        "--analysis_db_path", analysis_path,
        "--half" if half else ""]

    print(args)

    subprocess.run(args, cwd=os.getcwd())

def main(unused_argv):
    source    = FLAGS.source
    save_dir  = FLAGS.save_dir
    test_path = FLAGS.test_path

    with open(test_path) as f:
        tests = json.loads(f.read())

    # Extract the values from the JSON object
    values = [list(v) if isinstance(v, (list, tuple)) else [v]
              for v in tests.values()]

    # Generate all possible combinations of the values
    combinations = list(itertools.product(*values))

    # Print the combinations
    for c in combinations:
        imgsz, vid_stride, bs, model, tracker = c

        print(imgsz, vid_stride, model, tracker)
        
        analysis_fname = \
            f'{sanitise_fname(source)}_sz_{imgsz}_stride_{vid_stride}_model_' \
            f'{model.replace(".", "-")}_tracker_{tracker}.db'

        run_analysis(
            source,
            imgsz,
            vid_stride,
            model,
            tracker,
            analysis_path=str(Path(save_dir) / analysis_fname))

if __name__ == "__main__":
    app.run(main)