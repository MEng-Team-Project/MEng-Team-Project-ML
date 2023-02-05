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
import json

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
    """Get count and route analytical information of a video stream.
    
    args:
        stream  - Stream ID to get data for.
        raw     - (Optional) Provide the raw detection information in the result
        start   - (Optional) Start frame
        end     - (Optional) End frame
        classes - (Optional) List of COCO class labels to filter detections by
        trk_fmt - (Optional) Either `first_last` or `entire`. This will either
                  include the first and last anchor points for an object in the
                  route, or it will include the entire route for the requested
                  portion of the video. By default, returns `first_last`.

    """
    try:
        # Get stream ID
        content = request.json
        stream  = content["stream"]
        print("api/analysis->content:", content)

        # Get SQLite detection data
        analysis_path = Path(FLAGS.analysis_dir) / f"{stream}.db"
        con = sqlite3.connect(analysis_path)
        detections_df = pd.read_sql_query("SELECT * FROM detection;", con)
        con.close()

        # (Optional) Apply start and end frame filters
        data = detections_df
        if "start" in content:
            data = data[data["frame"] >= content["start"]]
        if "end" in content:
            data = data[data["frame"] <= content["end"]]
        if "classes" in content:
            classes = content["classes"]
            data    = data[data["label"].isin(classes)]
        
        # Get and extract count information for each (label, det_id) tuple
        counts = data.groupby('label')['det_id'].nunique().reset_index(name='count')

        # Get route information for each (label, det_id) tuple
        routes_df = data[["frame", "label", "det_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
        routes_df["anchor_x"] = routes_df.apply(
            lambda row: row["bbox_x"] + (row["bbox_w"] / 2.0), axis=1)
        routes_df["anchor_y"] = routes_df.apply(
            lambda row: row["bbox_y"] + (row["bbox_h"] / 2.0), axis=1)

        # Extract first and last or all anchor positions for each (label, det_id) tuple
        trk_fmt = "first_last"
        if "trk_fmt" in content:
            if content["trk_fmt"] == "entire":
                trk_fmt = "entire"
        def get_values(group, trk_fmt):
            vals = group[['anchor_x', 'anchor_y']].values.tolist()
            if trk_fmt == "first_last":
                vals = [vals[0], vals[-1]]
            vals = [{"x": val[0], "y": val[1]} for val in vals]
            return vals
        routes = routes_df.groupby(['label', 'det_id']).apply(
                    lambda group: get_values(group, trk_fmt))

        # Reset the index of the resulting series to remove the MultiIndex
        routes = routes.reset_index()

        # Replace the MultiIndex label column names with 'label' and 'det_id'
        routes.rename(columns={0: 'routes'}, inplace=True)
        routes.rename(columns={'level_0': 'label', 'level_1': 'det_id'}, inplace=True)

        # Create a dictionary with 'label' as the key and 'routes' as the value
        route_dict = routes.groupby('label')['routes'].apply(list).to_dict()

        # Separate raw data and analytical data
        final_data = {
            "counts": json.loads(counts.to_json(orient="records")),
            "routes": route_dict
        }

        if "raw" in content:
            if content["raw"] == True:
                final_data["raw"] = json.loads(data.to_json(orient="records"))

        return jsonify(final_data)
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