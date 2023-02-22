import time
import os
import subprocess

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
repeats = 3
cmd = lambda bs, trk, imgsz, exp: f'python yolov8-para.py --video_dir ./vids/ --batch_size {bs} {"--trk" if trk else ""} --model yolov8n.pt --imgsz {imgsz} --exp {exp}'

def experiment(exp, bs_s, trks, imgsz):
    for bs in bs_s:
        for trk in trks:
            for _ in range(repeats):
                for img in imgsz:
                    try:
                        cur = cmd(bs, trk, img, exp)
                        subprocess.run(cur.split(" "), cwd=os.getcwd(), stdout=subprocess.PIPE)
                    except subprocess.CalledProcessError as err:
                        print("err:", err)

if __name__ == "__main__":
    # Experiment 1: Downsize Image Cropping
    # Experiment 2: Batched Inference (Tracking-sep)
    experiments = [
        [
            "downsize",
            [1],
            [False],
            [320, 640]
        ],
        [
            "batched_track_combo",
            batch_sizes,
            [True],
            [640]
        ]
    ]

    for exp in experiments:
        experiment(*exp)