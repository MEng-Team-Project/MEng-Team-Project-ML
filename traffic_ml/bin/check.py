import json
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot

from absl import app
from absl import flags    

FLAGS = flags.FLAGS
flags.DEFINE_string ("host", "localhost", "Host IP")
flags.DEFINE_integer("port", 6000, "Host port")
flags.DEFINE_string ("analysis_dir", None, "Directory which yolo model is storing analysis outputs")

flags.mark_flag_as_required("analysis_dir")

def main(unused_argv):
    base_dir   = "C:\\Users\\win8t\\OneDrive\\Desktop\\projects\\traffic-core\\yolov7-segmentation\\runs\\predict-seg\\exp2\\labels\\"
    fname      = "SEM_ID_TRAFFIC_TEST_TILTON_TINY"
    info_path  = lambda idx: f"{fname}_info_{idx}.json"
    base_path  = lambda idx: f"{fname}_{idx}.json"
    track_path = lambda idx: f"{fname}_track_{idx}.json"

    a_s = [], b_s = [], c_s = []

    for i in range(1, 229+1):
        idx = i
        with open(os.path.join(base_dir, info_path(idx))) as f:
            content = f.readlines()
            obj = json.loads(content[0])
            a = len(obj["infos"])
            a_s.append(a)
            #print("info:", len(obj["infos"]))
            #print([info for info in obj["infos"]])
        with open(os.path.join(base_dir, base_path(idx))) as f:
            content = "[" + ",".join(f.readlines()) + "]"
            obj = json.loads(content)
            b = len(obj)
            b_s.append(b)
            #print("base:", len(obj))
            #print([info for info in obj])
        with open(os.path.join(base_dir, track_path(idx))) as f:
            content = f.readlines()
            obj = json.loads(content[0])
            c = len(obj["routes"])
            c_s.append(c)
            # print("routes:", len(obj["routes"]))
        print(i, a, b, c)

    ax = subplot(1,1,1)
    ax.plot(a_s, label="Tracked Detections")
    ax.plot(b_s, label="Segments")
    ax.plot(c_s, label="Routes")
    plt.legend()
    plt.show()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)