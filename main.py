import logging
import os
import subprocess

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

@app.route("/api/", methods=["GET"])
def index():
    """API Testing Endpoint."""
    return jsonify("Hello, World!")

@app.route("/api/analysis/", methods=["GET"])
def analysis():
    """Get analytical information of a video source.
    
    args:
        experiment - "exp" dir which contains analytical information.
                      Meant for debugging only."""
    try:
        args = request.args
        experiment = args.get("experiment")
        exp  = os.path.join(ANALYSIS_BASE, experiment, "/labels")
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