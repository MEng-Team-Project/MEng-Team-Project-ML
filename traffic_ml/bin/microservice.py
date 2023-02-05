# MIT License
# 
# Copyright (c) 2022-2023 MEng-Team-Project
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
from pathlib import Path
import pandas as pd
import sqlite3

from flask import Flask, request, jsonify

from absl import app as absl_app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string ("host", "localhost", "Host IP")
flags.DEFINE_integer("port", 6000, "Host port")
flags.DEFINE_string ("analysis_dir", None, "Directory to save analysis DBs to")

flags.mark_flag_as_required("analysis_dir")

app = Flask(__name__)

OFFLINE_ANALYSIS = lambda source, analysis_path, half: \
    f'python yolov8_tracking/track.py --source {source} --save-vid --save-trajectories --yolo-weights yolov8l.pt --tracking-method strongsort --analysis_db_path {analysis_path} {"--half" if half else ""}'

@app.route("/api/analysis/", methods=["POST"])
def analysis():
    """Get analytical information of a video stream.
    
    args:
        stream - Stream ID to get data for.
    """
    try:
        args          = request.args
        stream        = args.get("stream")
        analysis_path = Path(FLAGS.analysis_dir) / f"{stream}.db"
        con = sqlite3.connect(analysis_path)
        detections_df = pd.read_sql_query("SELECT * FROM detection;", con)
        data = detections_df.to_json(orient="records")
        con.close()
        return jsonify(data)
    except Exception as e:
        return jsonify("Error:", str(e)), 400

@app.route("/api/init", methods=["POST"])
def init():
    """Initiate an analysis of a video stream.
    
    args:
        stream - Absolute video stream path
    """
    try:
        content = request.json
        stream  = content["stream"]
        half    = True # Half-Precision Infer
        analysis_fname = Path(stream).stem
        analysis_path = Path(FLAGS.analysis_dir) / f"{analysis_fname}.db"
        args = list(filter(None, OFFLINE_ANALYSIS(stream, analysis_path, half).split(" ")))

        logging.info(args)

        # NOTE: Uncomment stdout and stderr for async operation, it's sync by default
        subprocess.run(
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