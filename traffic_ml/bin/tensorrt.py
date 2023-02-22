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
"""Uses a prebuilt TensorRT engine to perform inference with an ONNX format
YOLOv8 model."""

# Need to add YOLOv8-TensorRT submodule to PATH
import sys
sys.path.append('../../YOLOv8-TensorRT')

from models import TRTModule
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

import cv2
import torch

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("engine", None,   "TensorRT engine for inference (.engine)")
flags.DEFINE_string("device", "cuda", "Device to run TensorRT on")

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("engine")

def main(unused_argv):
    engine = FLAGS.engine
    device = torch.device(FLAGS.device)
    Engine = TRTModule(engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # Arrange desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

if __name__ == "__main__":
    app.run(main)