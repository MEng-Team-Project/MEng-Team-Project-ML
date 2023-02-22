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
"""Uses TensorRT for inference during production."""

"""
Required steps:
1. Convert pre-trained model into .onnx format
2. TensorRT (TRT) imported as py module (tensorrt as trt)
   - Batch size fixed on engine (TRT) creation
   This is then saved as an engine (if you are using it like this)
3. 
"""

from absl import app
from absl import flags

from ultralytics import YOLO

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "yolov8n.pt", "Model to convert to ONNX")

def main(unused_argv):
    model = YOLO(FLAGS.model)
    # msg   = model.export(format="onnx", opset=11)
    msg   = model.export(format="engine", device="0") # opset=11)
    # msg   = model.export(format="engine", opset=11)
    print(msg)

if __name__ == "__main__":
    app.run(main)