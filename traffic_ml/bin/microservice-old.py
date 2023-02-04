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
"""Traffic analysis deep learning microservice which performs analysis of
video footage for identification, count and route tracking of vehicles
and people."""

import logging
import os
import subprocess
import json

from flask import Flask, request, jsonify

from absl import app as absl_app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string ("host", "localhost", "Host IP")
flags.DEFINE_integer("port", 6000, "Host port")
flags.DEFINE_string ("analysis_dir", None, "Directory which yolo model is storing analysis outputs")

flags.mark_flag_as_required("analysis_dir")

app = Flask(__name__)

OFFLINE_ANALYSIS = lambda source, half, tracking: \
    f'python yolov7-segmentation/segment/predict.py --weights ./yolov7-seg.pt --source {source} --{half} --save-txt --save-conf --img-size 640 {tracking}'

def get_sub_analysis(base_dir, data_type, start=0, end=0):
    """Retrieves analysis data for specific data type for a video source."""
    print("start, end:", start, end)
    suffix = data_type if data_type else ""
    fi_s = list(filter(lambda x: suffix in x, os.listdir(base_dir)))
    fi_s = [os.path.join(base_dir, fi) for fi in fi_s]
    fi_s = sorted(fi_s, key=len)
    fi_s_s = []
    for i, fi in enumerate(fi_s):
        idx = int(fi.split("_")[-1].split(".")[0])
        if start > 0:
            if idx < start:
                continue
        if end > 0:
            if idx > end:
                continue
        with open(fi) as f:
            if data_type:
                data = "".join(list(filter(None, f.read().split("\n"))))
            else:
                data = "[" + ",".join(list(filter(None, f.read().split("\n")))) + "]"
            d = json.loads(data)
            d["frame"] = idx
        fi_s_s.append(d)
    data = json.dumps(fi_s_s)
    return data

def get_analysis(base_dir, start=0, end=0):
    """Retrieves all analysis data for a video source."""
    info_data    = get_sub_analysis(base_dir, "_info", start, end)
    track_data   = get_sub_analysis(base_dir, "_track", start, end)
    data = {
        "info":    info_data,
        "track":   track_data
    }
    return data

def get_experiment(stream):
    """Retrieves the most recent video analysis folder for a video source."""
    experiments = os.listdir(FLAGS.analysis_dir)
    found = [os.path.exists(
                os.path.join(FLAGS.analysis_dir, f"{exp_dir}/", f"{stream}.mp4"))
             for exp_dir in experiments]
    data = zip(experiments, found)
    data = [exp for exp, found in data if found]
    return data

@app.route("/api/download/", methods=["GET"])
def download():
    """Download all analytical information for a video source.
    
    args:
        stream      - Stream ID to get data for.
        destination - Local path to download file to."""
    try:
        args         = request.args
        stream       = args.get("stream")
        experiment   = get_experiment(stream)
        if len(experiment) == 0:
            return jsonify(f"Video has not been analysed: {stream}"), 400
        experiment   = experiment[-1]
        analysis_dir = os.path.join(FLAGS.analysis_dir, f"{experiment}/", "labels/")
        destination  = args.get("destination")
        if os.path.exists(destination):
            return jsonify("File already exists!")

        data = get_analysis(analysis_dir, 0, 0)
        with open(destination, "w") as f:
            f.write(json.dumps(data, indent=4))
        return jsonify(f"JSON data downloaded to: {destination}")
    except Exception as e:
        return jsonify("Error: " + str(e)), 400

@app.route("/api/analysis/", methods=["GET"])
def analysis():
    """Get analytical information of a video source.
    
    args:
        stream     - Stream ID to get data for.
        data_type  - Data type to retrieve (info, track).
        start      - (Optional) Start frame to retrieve data from, inclusive.
        end        - (Optional) End frame to retrieve data to, inclusive."""
    try:
        args         = request.args
        stream       = args.get("stream")
        experiment   = get_experiment(stream)
        experiment   = experiment[-1]
        data_type    = args.get("data_type")
        start        = args.get("start") if "start" in args else -1
        end          = args.get("end")   if "end" in args else -1
        analysis_dir = os.path.join(FLAGS.analysis_dir, f"{experiment}/", "labels/")

        if data_type == "all":
            data = get_analysis(analysis_dir, int(start), int(end))
        else:
            data = get_sub_analysis(analysis_dir, data_type, int(start), int(end))

        return jsonify(data)
    except Exception as e:
        return jsonify("Error:", str(e)), 400

@app.route("/api/init", methods=["POST"])
def init():
    """Initiate an analysis of a video source.
    
    args:
        source   - Absolute video stream source path
        type     - "online"/"offline" analysis"""
    try:
        content  = request.json
        source   = content["source"]
        half     = True
        half     = "half" if half else ""
        tracking = True
        tracking = "--trk" if tracking else ""

        args = list(filter(None, OFFLINE_ANALYSIS(source, half, tracking).split(" ")))

        logging.info(args)
        subprocess.run(
            args,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            cwd=os.getcwd())

        return jsonify("Video stream analysis successfully started")

    except Exception as e:
        return jsonify("Error:", str(e)), 400

@app.route("/api/", methods=["GET"])
def get():
    return jsonify("Hiya!")

def main(unused_argv):
    app.run(host=FLAGS.host, port=FLAGS.port)

def entry_point():
    absl_app.run(main)

if __name__ == "__main__":
    absl_app.run(main)