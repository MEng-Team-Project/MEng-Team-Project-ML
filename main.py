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

def get_analysis(base_dir):
    info_s = list(filter(lambda x: "_info" in x, os.listdir(base_dir)))
    info_s = [os.path.join(base_dir, info) for info in info_s]
    info_s = sorted(info_s, key=len)
    info_s_s = []
    for i, info in enumerate(info_s):
        with open(info) as f:
            data = "".join(list(filter(None, f.read().split("\n"))))
            d = json.loads(data)
        info_s_s.append(d)
    data = json.dumps(info_s_s)
    return data

@app.route("/api/analysis/", methods=["GET"])
def analysis():
    """Get analytical information of a video source.
    
    args:
        experiment - "exp" dir which contains analytical information.
                      Meant for debugging only."""
    try:
        args         = request.args
        experiment   = args.get("experiment")
        stream       = args.get("stream")
        analysis_dir = os.path.join(ANALYSIS_BASE, f"{experiment}/", "labels/")
        data         = get_analysis(analysis_dir)
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