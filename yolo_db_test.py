import os
import json
import sqlite3

def insert_base(
    cur,
    base_count,
    frame,
    class_idx,
    x1,
    y1,
    x2,
    y2,
    conf):
    cmd = f"""INSERT INTO detections VALUES (
        {base_count},
        {frame},
        {class_idx},
        {x1},
        {y1},
        {x2},
        {y2},
        {conf}
    )"""
    cur.execute(cmd)

def insert_info(
    cur,
    tracked_count,
    frame,
    bbox_x1,
    bbox_y1,
    bbox_x2,
    bbox_y2,
    label,
    anchor_x,
    anchor_y):
    cmd = f"""INSERT INTO tracked VALUES (
        {tracked_count},
        {frame},
        {bbox_x1},
        {bbox_y1},
        {bbox_x2},
        {bbox_y2},
        {label},
        {anchor_x},
        {anchor_y}
    )"""
    cur.execute(cmd)

def insert_route(
    cur,
    route_count,
    frame,
    route_idx):
    cmd = f"""INSERT INTO routes VALUES (
        {route_count},
        {frame},
        {route_idx}
    )"""
    cur.execute(cmd)

def insert_sub_track(
    cur,
    subroute_count,
    route_id,
    x1,
    y1,
    x2,
    y2):
    cmd = f"""INSERT INTO route VALUES (
        {subroute_count},
        {route_id},
        {x1},
        {y1},
        {x2},
        {y2}
    )"""
    cur.execute(cmd)

if __name__ == "__main__":
    # Delete Old on Init
    if os.path.exists("analysis.db"):
        os.remove("analysis.db")

    # Paths
    base_dir   = "./yolov7-segmentation/runs/predict-seg/exp/labels/"
    target_idx = 100
    target     = "00001.01350_2022-12-07T15-35-24.000Z"
    fname      = lambda target_idx, suffix:\
        f"{target}_{suffix}{target_idx}"

    # Base
    with open(os.path.join(base_dir, fname(target_idx, "") + ".json")) as f:
        base = list(filter(None, f.read().split("\n")))
        base = "[" + ",".join(base) + "]"
        base = json.loads(base)
    
    # Info
    with open(os.path.join(base_dir, fname(target_idx, "info_") + ".json")) as f:
        info = f.read()
        info = json.loads(info)["infos"]
    
    # Track
    with open(os.path.join(base_dir, fname(target_idx, "track_") + ".json")) as f:
        tracks = f.read()
        tracks = json.loads(tracks)["routes"]

    # Create SQL Tables
    con = sqlite3.connect("./analysis.db")
    cur = con.cursor()
    with open("./yolov7-segmentation/segment/CREATE_TABLES.sql") as f:
        sql_commands = f.read().split(";")
        for sql_command in sql_commands:
            command = sql_command + ";"
            print(command)
            cur.execute(command)

    # Insert Testing Analytics Data
    cur.execute("BEGIN;")

    # Per Frame Inserts
    base_count     = 0
    tracked_count  = 0
    route_count    = 0
    subroute_count = 0
    for frame in [target_idx]: # [2] will instead be list of frames in YoloV7 from [1, ..., frames]
        for base_item in base:
            insert_base(
                cur,
                base_count,
                frame,
                base_item["class"],
                base_item["x1"],
                base_item["y1"],
                base_item["x2"],
                base_item["y2"],
                base_item["conf"])
            base_count += 1

        for info_item in info:
            insert_info(
                cur,
                tracked_count,
                frame,
                info_item["bbox"]["x1"],
                info_item["bbox"]["y1"],
                info_item["bbox"]["x2"],
                info_item["bbox"]["y2"],
                info_item["label"],
                info_item["anchor"]["x"],
                info_item["anchor"]["y"])
            tracked_count += 1

        for route_idx, route in enumerate(tracks):
            frame = frame
            insert_route(
                cur,
                route_count,
                frame,
                route_idx)
            
            for subroute in route:
                insert_sub_track(
                    cur,
                    subroute_count,
                    route_count,
                    subroute["x1"],
                    subroute["y1"],
                    subroute["x2"],
                    subroute["y2"])
                subroute_count += 1

            route_count += 1

    cur.execute("COMMIT;")
    con.close()