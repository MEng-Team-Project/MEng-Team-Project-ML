import logging
import os
import subprocess
import json

from flask import Flask, request, jsonify

from absl import app as absl_app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string ("host", "localhost", "Host IP")
flags.DEFINE_integer("port", 6000,        "Host port")

app = Flask(__name__)

ANALYSIS_BASE = "C://Users//win8t//OneDrive//Desktop//projects//traffic-core//yolov7-segmentation//runs//predict-seg//"
OFFLINE_ANALYSIS = lambda source, half, tracking: \
    f'python yolov7-segmentation/segment/predict.py --weights ./yolov7-seg.pt --source {source} --{half} --save-txt --save-conf --img-size 640 {tracking}'

def get_sub_analysis(base_dir, data_type, start=0, end=0):
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
    info_data    = get_sub_analysis(base_dir, "_info", start, end)
    track_data   = get_sub_analysis(base_dir, "_track", start, end)
    data = {
        "info":    info_data,
        "track":   track_data
    }
    return data

@app.route("/api/analysis/", methods=["GET"])
def analysis():
    """Get analytical information of a video source.
    
    args:
        experiment - "exp" dir which contains analytical information.
                      Meant for debugging only.
        stream     - Stream ID to get data for.
        data_type  - Data type to retrieve (info, track).
        start      - (Optional) Start frame to retrieve data from, inclusive.
        end        - (Optional) End frame to retrieve data to, inclusive."""
    try:
        args         = request.args
        experiment   = args.get("experiment")
        stream       = args.get("stream")
        data_type    = args.get("data_type")
        start        = args.get("start") if "start" in args else -1
        end          = args.get("end")   if "end" in args else -1
        analysis_dir = os.path.join(ANALYSIS_BASE, f"{experiment}/", "labels/")
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
        subprocess.Popen(
            args,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            cwd=os.getcwd())

        return jsonify("Video stream analysis successfully started")

    except Exception as e:
        return jsonify("Error:", str(e)), 400

def main(unused_argv):
    app.run(host=FLAGS.host, port=FLAGS.port)

def entry_point():
    absl_app.run(main)

if __name__ == "__main__":
    absl_app.run(main)