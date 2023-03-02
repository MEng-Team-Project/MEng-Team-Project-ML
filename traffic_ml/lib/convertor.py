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
"""Convert video annotation files into pd.DataFrame format used in this
project."""

import json
import pandas as pd


class Annotations(object):
    @staticmethod
    def from_darwin(darwin_fname):
        with open(darwin_fname) as f:
            obj = json.loads(f.read())
            version = obj["version"]
            if version != "2.0":
                return None
            
            # Contains {} for each tracked object
            annotations = obj["annotations"]

            entries = []

            for i, annotation in enumerate(annotations):
                frames  = annotation["frames"]
                frame_idxs = frames.keys()
                cls_lbl = annotation["name"]

                for frame_idx in frame_idxs:
                    frame = frames[frame_idx]
                    bbox  = frame["bounding_box"]
                    h     = float(int(bbox["h"]))
                    w     = float(int(bbox["w"]))
                    x     = float(int(bbox["x"]))
                    y     = float(int(bbox["y"]))
                    entry = [int(frame_idx), x, y, w, h, cls_lbl, 1.0, float(i)]
                    entries.append(entry)

            df = pd.DataFrame(entries, columns=[
                "frame",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "label",
                "conf",
                "det_id"])

            df = df.sort_values(by=["frame"])
            return df