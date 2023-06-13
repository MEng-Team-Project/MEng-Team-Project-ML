<<<<<<< HEAD
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
import numpy as np

from shapely.geometry import Point, Polygon

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

@app.route("/api/routes/", methods=["POST"])
def routes():
    """Get per object count for each supplied route region
    
    args:
        stream  - Stream ID to get data for.
        regions - Route region polygon information
        start   - (Optional) Start frame
        end     - (Optional) End frame
        classes - (Optional) List of COCO class labels to filter detections by
    """
    try:
        # Get stream ID
        content = request.json
        stream  = content["stream"]
        print("api/routes->content:", content)

        # Check route info
        if not "regions" in content:
            return jsonify("Error: Route region polygons required"), 400

        # Get SQLite detection data
        analysis_path = Path(FLAGS.analysis_dir) / f"{stream}.db"
        con = sqlite3.connect(analysis_path)
        detections_df = pd.read_sql_query("SELECT * FROM detection;", con)
        metadata_df   = pd.read_sql_query("SELECT * FROM metadata;", con)
        metadata_df   = metadata_df.iloc[0]
        if "id" in metadata_df:
            del metadata_df["id"]
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
        
        # NOTE: SAME TILL HERE
        routes_df = data[["frame", "label", "det_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
        routes_df["anchor_x"] = routes_df.apply(
            lambda row: row["bbox_x"] + (row["bbox_w"] / 2.0), axis=1)
        routes_df["anchor_y"] = routes_df.apply(
            lambda row: row["bbox_y"] + (row["bbox_h"] / 2.0), axis=1)

        def get_values(group):
            return group[['anchor_x', 'anchor_y']].values.tolist()

        entire_routes = routes_df.groupby(['label', 'det_id']).apply(get_values)


        import json
        trk_fmt = "first_last"

        def get_values(group, trk_fmt):
            vals = group[['frame', 'anchor_x', 'anchor_y']].values.tolist()
            if trk_fmt == "first_last":
                vals = [vals[0], vals[-1]]
            vals = [{"frame:": val[0], "x": val[1], "y": val[2]} for val in vals]
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

        # 1. Get start and end pos for each unique object during entire video
        start_end_df = entire_routes.copy()
        start_end_df = start_end_df.reset_index()

        overlap_start_df = start_end_df.copy()[["label", "det_id"]]
        overlap_end_df = start_end_df.copy()[["label", "det_id"]]

        start_end_df["start_x"] = start_end_df.apply(lambda row: row[0][0][0],  axis=1)
        start_end_df["start_y"] = start_end_df.apply(lambda row: row[0][0][1],  axis=1)
        start_end_df["end_x"]   = start_end_df.apply(lambda row: row[0][-1][0], axis=1)
        start_end_df["end_y"]   = start_end_df.apply(lambda row: row[0][-1][1], axis=1)
        if 0 in start_end_df:
            del start_end_df[0]

        # 2. Get overlaps between start and end, and each region
        ROUTE_REGIONS = content["regions"]
        for route_region in ROUTE_REGIONS.keys():
            cur_polygon = Polygon(ROUTE_REGIONS[route_region])

            # Start Geometry
            start_geometry = [Point(x, y) for x, y in zip(start_end_df["start_x"], start_end_df["start_y"])]

            # Start Overlap
            start_end_df[f"{route_region}_start_overlap"] = [point.within(cur_polygon) for point in start_geometry]
            overlap_start_df[route_region]                = start_end_df[f"{route_region}_start_overlap"]

            # End Geometry
            end_geometry = [Point(x, y) for x, y in zip(start_end_df["end_x"], start_end_df["end_y"])]

            # End Overlap
            start_end_df[f"{route_region}_end_overlap"] = [point.within(cur_polygon) for point in end_geometry]
            overlap_end_df[route_region]                = start_end_df[f"{route_region}_end_overlap"]

        def label_overlap(row):
            for i in range(len(row)):
                if row[i]:
                    return overlap_start_df.columns[2:][i]
            return np.nan

        overlap_start_df['overlap_label'] = overlap_start_df.iloc[:, 2:].apply(label_overlap, axis=1)
        overlap_end_df['overlap_label'] = overlap_end_df.iloc[:, 2:].apply(label_overlap, axis=1)

        overlap_df = start_end_df.iloc[:, :2]
        overlap_df["start"] = overlap_start_df["overlap_label"]
        overlap_df["end"]   = overlap_end_df["overlap_label"]

        overlap = json.loads(overlap_df.to_json(orient="records"))
        return jsonify(overlap)

    except Exception as e:
        return jsonify("Error:", str(e)), 400

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
        metadata_df   = pd.read_sql_query("SELECT * FROM metadata;", con)
        metadata_df   = metadata_df.iloc[0]
        if "id" in metadata_df:
            del metadata_df["id"]
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
        counts_df = data.groupby('label')['det_id'].nunique().reset_index(name='count')
        counts    = json.loads(counts_df.to_json(orient="records"))

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
            vals = group[['frame', 'anchor_x', 'anchor_y']].values.tolist()
            if trk_fmt == "first_last":
                vals = [vals[0], vals[-1]]
            vals = [{"frame:": val[0], "x": val[1], "y": val[2]} for val in vals]
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

        # Convert metadata into dict
        metadata = json.loads(metadata_df.to_json(orient="index"))

        # Separate raw data and analytical data
        final_data = {
            "metadata": metadata,
            "tracking_format": trk_fmt,
            "counts": counts,
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
=======
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
import numpy as np

from shapely.geometry import Point, Polygon

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

@app.route("/api/routes/", methods=["POST"])
def routes():
    """Get per object count for each supplied route region
    
    args:
        stream  - Stream ID to get data for.
        regions - Route region polygon information
        start   - (Optional) Start frame
        end     - (Optional) End frame
        classes - (Optional) List of COCO class labels to filter detections by
    """
    try:
        # Get stream ID
        content = request.json
        stream  = content["stream"]
        print("api/routes->content:", content)

        # Check route info
        if not "regions" in content:
            return jsonify("Error: Route region polygons required"), 400

        # Get SQLite detection data
        analysis_path = Path(FLAGS.analysis_dir) / f"{stream}.db"
        con = sqlite3.connect(analysis_path)
        detections_df = pd.read_sql_query("SELECT * FROM detection;", con)
        metadata_df   = pd.read_sql_query("SELECT * FROM metadata;", con)
        metadata_df   = metadata_df.iloc[0]
        if "id" in metadata_df:
            del metadata_df["id"]
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
        
        # NOTE: SAME TILL HERE
        routes_df = data[["frame", "label", "det_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
        routes_df["anchor_x"] = routes_df.apply(
            lambda row: row["bbox_x"] + (row["bbox_w"] / 2.0), axis=1)
        routes_df["anchor_y"] = routes_df.apply(
            lambda row: row["bbox_y"] + (row["bbox_h"] / 2.0), axis=1)

        def get_values(group):
            return group[['anchor_x', 'anchor_y']].values.tolist()

        entire_routes = routes_df.groupby(['label', 'det_id']).apply(get_values)


        import json
        trk_fmt = "first_last"

        def get_values(group, trk_fmt):
            vals = group[['frame', 'anchor_x', 'anchor_y']].values.tolist()
            if trk_fmt == "first_last":
                vals = [vals[0], vals[-1]]
            vals = [{"frame:": val[0], "x": val[1], "y": val[2]} for val in vals]
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

        # 1. Get start and end pos for each unique object during entire video
        start_end_df = entire_routes.copy()
        start_end_df = start_end_df.reset_index()

        overlap_start_df = start_end_df.copy()[["label", "det_id"]]
        overlap_end_df = start_end_df.copy()[["label", "det_id"]]

        start_end_df["start_x"] = start_end_df.apply(lambda row: row[0][0][0],  axis=1)
        start_end_df["start_y"] = start_end_df.apply(lambda row: row[0][0][1],  axis=1)
        start_end_df["end_x"]   = start_end_df.apply(lambda row: row[0][-1][0], axis=1)
        start_end_df["end_y"]   = start_end_df.apply(lambda row: row[0][-1][1], axis=1)
        if 0 in start_end_df:
            del start_end_df[0]

        # 2. Get overlaps between start and end, and each region
        ROUTE_REGIONS = content["regions"]
        for route_region in ROUTE_REGIONS.keys():
            cur_polygon = Polygon(ROUTE_REGIONS[route_region])

            # Start Geometry
            start_geometry = [Point(x, y) for x, y in zip(start_end_df["start_x"], start_end_df["start_y"])]

            # Start Overlap
            start_end_df[f"{route_region}_start_overlap"] = [point.within(cur_polygon) for point in start_geometry]
            overlap_start_df[route_region]                = start_end_df[f"{route_region}_start_overlap"]

            # End Geometry
            end_geometry = [Point(x, y) for x, y in zip(start_end_df["end_x"], start_end_df["end_y"])]

            # End Overlap
            start_end_df[f"{route_region}_end_overlap"] = [point.within(cur_polygon) for point in end_geometry]
            overlap_end_df[route_region]                = start_end_df[f"{route_region}_end_overlap"]

        def label_overlap(row):
            for i in range(len(row)):
                if row[i]:
                    return overlap_start_df.columns[2:][i]
            return np.nan

        overlap_start_df['overlap_label'] = overlap_start_df.iloc[:, 2:].apply(label_overlap, axis=1)
        overlap_end_df['overlap_label'] = overlap_end_df.iloc[:, 2:].apply(label_overlap, axis=1)

        overlap_df = start_end_df.iloc[:, :2]
        overlap_df["start"] = overlap_start_df["overlap_label"]
        overlap_df["end"]   = overlap_end_df["overlap_label"]

        overlap = json.loads(overlap_df.to_json(orient="records"))
        return jsonify(overlap)

    except Exception as e:
        return jsonify("Error:", str(e)), 400

@app.route("/api/routeAnalytics/", methods=["POST"])
def routeAnalytics():
    """ Aggregate analytic data from specific database file and filtered for
        frontend 
    args:
        stream              - Stream ID to get data for.
        regions             - Route region polygon information
        classes             - List of COCO class labels to filter by
        time_of_recording   - UTC float start time of source recording
        start_time          - start of analytics collection
        end_time            - end of analytics collection (default to end of source) 
        start_regions       - (Optional) List of start regions to filter by (default all)
        end_regions         - (Optional) List of end regions to filter by (default all)
        interval_spacing    - (Optional) Interval spacing to split up detections by (secs)
        fps                 - (Optional) FPS to timestamp each frame
    """
    try:
        import arrow 
        import sqlite3
        import pandas as pd
        import numpy as np
        from shapely.geometry import Point, Polygon

        # Get stream ID
        content = request.json
        stream  = content["stream"]
        print("api/routeAnalytics->content:", content)

        # Check required fields
        args = ['stream', 'regions', 'classes', 'start_time']
        for arg in args:
            if arg not in content:
                return jsonify(f"Error: {arg.title()} required"), 400

        # Main constant assignments
        FILE_NAME           = stream
        ROUTE_REGIONS       = content['regions']
        CLASSES             = content['classes']
        TIME_OF_RECORDING   = arrow.get(content['time_of_recording']) # UTC to Arrow
        START_TIME          = arrow.get(content['start_time'])
        if 'end_time' in content:
            END_TIME        = arrow.get(content['end_time'])
        if 'start_regions' in content:
            START_REGIONS   = content['start_regions']
        else:
            START_REGIONS   = ROUTE_REGIONS.keys() 
        if 'end_regions' in content:
            END_REGIONS     = content['end_regions']
        else:
            END_REGIONS     = ROUTE_REGIONS.keys()
        if 'interval_spacing' in content:
            INTERVAL_SPACING    = content['interval_spacing']
        else:
            INTERVAL_SPACING = None


        # Get SQLite detection data
        analysis_path = Path(FLAGS.analysis_dir) / f"{stream}.db"
        print(analysis_path.absolute())
        con = sqlite3.connect(analysis_path)
        
        detections_df = pd.read_sql_query("SELECT * FROM detection;", con)
        detections_df = detections_df.reindex(columns=['frame', 'label', 'det_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])

        ## Get FPS from metadata
        if 'fps' not in content:
            metadata_df = pd.read_sql_query("SELECT * FROM metadata;", con)
            metadata_df = metadata_df.iloc[0]
            if 'fps' in metadata_df:
                FPS = metadata_df['fps']
            else:
                FPS = 30 # Default FPS value
            
        con.close()
                
        ### Detection Count (per Frame)
        ### Change Frames to Timestamps
        def newTimestamp(timestamp, frame, fps):
            return timestamp.shift(microseconds=1000000/fps * frame)

        # Change frames to timestamps
        detections_df['frame'] = detections_df.apply(
            lambda row: newTimestamp(TIME_OF_RECORDING, int(row["frame"]), FPS), 
            axis=1
        )
        detections_df = detections_df.rename(columns={'frame':'timestamp'})
        data = detections_df

        ### Object Count
        ### Object Tracking
        routes_df = data[["timestamp", "label", "det_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
        routes_df["anchor_x"] = routes_df.apply(
            lambda row: row["bbox_x"] + (row["bbox_w"] / 2.0), axis=1)
        routes_df["anchor_y"] = routes_df.apply(
            lambda row: row["bbox_y"] + (row["bbox_h"] / 2.0), axis=1)

        #### Either First and Last, or all Anchor Points for Each (label, det_id) Tuple
        def get_values(group):
            return group[['anchor_x', 'anchor_y']].values.tolist()

        entire_routes = routes_df.groupby(['label', 'det_id']).apply(get_values)
        ### Object Tracking (Start and Finish Regions)
        # 1. Get start and end pos for each unique object during entire video
        start_end_df = entire_routes.copy()
        start_end_df = start_end_df.reset_index()

        overlap_start_df = start_end_df.copy()[["label", "det_id"]]
        overlap_end_df = start_end_df.copy()[["label", "det_id"]]

        start_end_df["start_x"] = start_end_df.apply(lambda row: row[0][0][0],  axis=1)
        start_end_df["start_y"] = start_end_df.apply(lambda row: row[0][0][1],  axis=1)
        start_end_df["end_x"]   = start_end_df.apply(lambda row: row[0][-1][0], axis=1)
        start_end_df["end_y"]   = start_end_df.apply(lambda row: row[0][-1][1], axis=1)
        if 0 in start_end_df:
            del start_end_df[0]

        # 2. Get overlaps between start and end, and each region
        region_polys = {}
        for route_region in ROUTE_REGIONS.keys():
            cur_polygon = Polygon(ROUTE_REGIONS[route_region])
            region_polys[route_region] = cur_polygon

            # Start Geometry
            start_geometry = [Point(x, y) for x, y in zip(start_end_df["start_x"], start_end_df["start_y"])]

            # Start Overlap
            start_end_df[f"{route_region}_start_overlap"] = [point.within(cur_polygon) for point in start_geometry]
            overlap_start_df[route_region]                = start_end_df[f"{route_region}_start_overlap"]

            # End Geometry
            end_geometry = [Point(x, y) for x, y in zip(start_end_df["end_x"], start_end_df["end_y"])]

            # End Overlap
            start_end_df[f"{route_region}_end_overlap"] = [point.within(cur_polygon) for point in end_geometry]
            overlap_end_df[route_region]                = start_end_df[f"{route_region}_end_overlap"]

        #### Determine Start Label
        def label_overlap(row):
            for i in range(len(row)):
                if row[i]:
                    return overlap_start_df.columns[2:][i]
            return np.nan

        overlap_start_df['overlap_label'] = overlap_start_df.iloc[:, 2:].apply(label_overlap, axis=1)

        #### Determine End Label
        def label_overlap(row):
            for i in range(len(row)):
                if row[i]:
                    return overlap_end_df.columns[2:][i]
            return np.nan

        overlap_end_df['overlap_label'] = overlap_end_df.iloc[:, 2:].apply(label_overlap, axis=1)

        #### Create Final Overlap DataFrame
        overlap_df = start_end_df.iloc[:, :2]
        overlap_df["start"] = overlap_start_df["overlap_label"]
        overlap_df["end"]   = overlap_end_df["overlap_label"]

        ### Finding Times Spent in and Out of Regions
        #### Organise Route Boundary DataFrame
        def get_values(group):
            return group[['timestamp','anchor_x', 'anchor_y']].values.tolist()

        route_boundaries_df = routes_df.groupby(['label', 'det_id']).apply(get_values)
        route_boundaries_df = pd.DataFrame(route_boundaries_df.reset_index(name='route'))

        # Add region names from overlap_df
        route_boundaries_df = pd.merge(route_boundaries_df, overlap_df, on=['label', 'det_id'])
        route_boundaries_df.rename(columns={'start':'start_region', 'end':'end_region'}, inplace=True)
        entire_routes = route_boundaries_df.copy()

        #### Arrange Start and End Points
        route_boundaries_df['start_point'] = route_boundaries_df['route'].apply(lambda route: route[0])
        route_boundaries_df['end_point'] = route_boundaries_df['route'].apply(lambda route: route[-1])

        #### Find Starting Region Boundary
        def findFirstRegionBoundary(vals):
            initialRegion = which_region(Point(*vals[0][1:]))
            if initialRegion is None: return None
            
            # Binary search for boundary point
            pointRange = vals
            while len(pointRange) > 1:
                midIndex = round(len(pointRange)/2)
                midpoint = pointRange[midIndex]
                regionCheck = which_region(Point(*midpoint[1:]))

                if regionCheck == initialRegion:
                    pointRange = pointRange[midIndex:] # go right
                else:
                    pointRange = pointRange[:midIndex] # go left
            
            if len(pointRange) != 0: return pointRange[0]

        def which_region(point):
            for label, poly in region_polys.items():
                if point.within(poly):
                    return label

        route_boundaries_df['start_boundary'] = route_boundaries_df['route'].apply(findFirstRegionBoundary)

        #### Find Ending Region Boundary
        def findEndRegionBoundary(start_region, end_region, route):
            if start_region == end_region:
                return None
            else:
                return findFirstRegionBoundary(route[::-1])

        # Reverse the list and find the end region boundary
        route_boundaries_df['end_boundary'] = route_boundaries_df.apply(
            lambda row: findEndRegionBoundary(row['start_region'], row['end_region'], row['route']),
            axis=1
        )

        #### Organise Dataframe
        # Reorder columns
        route_boundaries_df = route_boundaries_df.reindex(columns=['label', 'det_id', 'start_region', 'start_point', 'start_boundary', 'end_boundary', 'end_point', 'end_region'])
        route_boundaries_df.loc[route_boundaries_df['label'] == 'car']

        # ROUTE BOUNDARIES FINISHED HERE #

        #### Workout Times Spent in Each Region From Timestamps
        route_times_df = route_boundaries_df.copy()

        def find_time(start_boundary, end_boundary):
            if start_boundary is None or end_boundary is None: 
                return None
            else:
                return end_boundary[0] - start_boundary[0]

        # Time spent overall in the route
        route_times_df['overall_time'] = route_times_df.apply(
            lambda row: row['end_point'][0] - row['start_point'][0], 
            axis=1
        )

        # Time spent in first region
        route_times_df['start_region_time'] = route_times_df.apply(
            lambda row: find_time(row['start_point'], row['start_boundary']), 
            axis=1
        )

        # Time spent in last region
        route_times_df['end_region_time'] = route_times_df.apply(
            lambda row: row['start_region_time'] if pd.isna(result := find_time(row['end_boundary'], row['end_point'])) else result,
            axis=1
        )

        def out_of_region_time(overall_time, start_region_time, end_region_time):
            startIsNotNone = not pd.isna(start_region_time)
            endIsNotNone = not pd.isna(end_region_time)

            # Ensure if there's only one region the no_region_time is None
            if overall_time == start_region_time: return None

            if startIsNotNone and endIsNotNone and start_region_time != end_region_time:
                return (overall_time - start_region_time) - end_region_time
            elif startIsNotNone:
                return overall_time - start_region_time
            elif endIsNotNone:
                return overall_time - end_region_time
            else:
                return overall_time
            
        # Time spent not in a region
        route_times_df['no_region_time'] = route_times_df.apply(
            lambda row: out_of_region_time(row['overall_time'], row['start_region_time'], row['end_region_time']), 
            axis=1
        )

        # Reorder columns
        route_times_df = route_times_df.reindex(columns=['label', 'det_id', 'start_region', 'overall_time', 'start_region_time', 'end_region_time', 'no_region_time', 'end_region'])

        ## Add start and end times of detection for time filtering
        route_times_df['start_time'] = route_boundaries_df['start_point'].apply(
            lambda row: row[0],
        )
        route_times_df['end_time'] = route_boundaries_df['end_point'].apply(
            lambda row: row[0],
        )

        ### Sort DataFrame
        route_times_df.sort_values(['start_time', 'end_time'], axis=0, inplace=True)

        ### Data filtering
        route_times_df = route_times_df[route_times_df['label'].isin(CLASSES)]

        ### Splitting the detections by timestamp intervals
        if 'end_time' not in content:
            END_TIME = route_times_df['end_time'].iloc[-1]

        if INTERVAL_SPACING is None:
            INTERVAL_SPACING = END_TIME - TIME_OF_RECORDING
            INTERVAL_SPACING = INTERVAL_SPACING.days * 86400 \
                               + INTERVAL_SPACING.seconds
            INTERVAL_SPACING = 30 if INTERVAL_SPACING == 0 else INTERVAL_SPACING

        timeBoundaries = [i for i in arrow.Arrow.interval('seconds', START_TIME, END_TIME, INTERVAL_SPACING)]
        BoundaryEnds = [i[1] for i in timeBoundaries]

        detSplit = [[] for i in range(len(BoundaryEnds))]
        currentDet = 0
        for boundaryIndex, timeBoundary in enumerate(timeBoundaries):
            nextBound = False

            for detIndex, det in route_times_df.iloc[currentDet:].iterrows():
                start = det['start_time']
                end = det['end_time']

                if start.is_between(*timeBoundary):
                    # if end within time interval or at least has more time in interval
                    if end.is_between(*timeBoundary) or \
                            timeBoundary[1] - start > end - timeBoundary[1] or \
                            boundaryIndex + 1 >= len(detSplit): 
                        destInterval = boundaryIndex # Add detection to current interval
                    else:
                        destInterval = boundaryIndex + 1 # Add detection to next interval
                    detSplit[destInterval].append(det)
                else:
                    nextBound = True
                    break

                currentDet += 1

            if nextBound:
                continue

        ### Split Detections into data-structure with interval stamps

        # Separate Detections (dets) by period of time
        countsAtTimes = [{'periodFrom'  : from_.float_timestamp, 
                        'periodTo'    : to_.float_timestamp,
                        'routeCounts' : pd.DataFrame(dets, columns=route_times_df.columns)} \
                    for (from_, to_), dets in zip(timeBoundaries, detSplit)]
        countsAtTimes[0]['routeCounts']

        # Count detection types by their label and sum their counts in 'total'
        def countByClass(df):
            counts = {'total':0}
            for _, row in df.iterrows():
                label = row['label']
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1
                counts['total'] += 1
            return counts

        # Separate Further into Directional Combinations of Start region and End region for each interval
        to_remove = []
        for index, interval in enumerate(countsAtTimes):
            routeCounts = interval['routeCounts']
            
            # Filter detections by their Start and End regions
            routeCounts = routeCounts[
                (routeCounts['start_region'].isin(START_REGIONS)) &
                (routeCounts['end_region'].isin(END_REGIONS))]
            
            # Split detections by start end region combinations
            detSplit = {}
            for _, det in routeCounts.iterrows():
                start, end = det['start_region'], det['end_region']
                if start == end:
                    continue # filter out stationary detections
                elif (start, end) not in detSplit:
                    detSplit[(start, end)] = [det]
                else:
                    detSplit[(start, end)].append(det)

            # Structure the split detections
            routeCounts = []
            for key, value in detSplit.items():
                df = pd.DataFrame(value, columns=route_times_df.columns)
                routeCounts.append({
                    'start': key[0], 
                    'end': key[1], 
                    'counts': countByClass(df)
                })

            # If routes empty add list to delete time period
            if len(routeCounts) == 0:
                to_remove.append(index)

            interval['routeCounts'] = routeCounts

        # Remove empty routeCounts
        temp_list = []
        for i, interval in enumerate(countsAtTimes):
            if i not in to_remove:
                temp_list.append(interval)
        countsAtTimes = temp_list

        # Structure the rest of the json message
        final_data = {
            "dataSource": FILE_NAME,
            "regions": list(ROUTE_REGIONS.keys()),
            "countsAtTimes": countsAtTimes
        }
        # print(json.dumps(final_data, indent=4))
        return jsonify(final_data)

    except Exception as e:
        return jsonify("Error:", str(e)), 400

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
        metadata_df   = pd.read_sql_query("SELECT * FROM metadata;", con)
        metadata_df   = metadata_df.iloc[0]
        if "id" in metadata_df:
            del metadata_df["id"]
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
        counts_df = data.groupby('label')['det_id'].nunique().reset_index(name='count')
        counts    = json.loads(counts_df.to_json(orient="records"))

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
            vals = group[['frame', 'anchor_x', 'anchor_y']].values.tolist()
            if trk_fmt == "first_last":
                vals = [vals[0], vals[-1]]
            vals = [{"frame:": val[0], "x": val[1], "y": val[2]} for val in vals]
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

        # Convert metadata into dict
        metadata = json.loads(metadata_df.to_json(orient="index"))

        # Separate raw data and analytical data
        final_data = {
            "metadata": metadata,
            "tracking_format": trk_fmt,
            "counts": counts,
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
>>>>>>> origin/analytics
    absl_app.run(main)