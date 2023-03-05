# MEng-Team-Project-ML

MEng-Team-Project-ML contains the microservice which performs analysis 
of recorded and live video streams to extract information relating to 
the identity, count and travelled route of motor vehicles, bikes and 
people. This repo also contains notebooks with analysis of real data 
extracted using real-world datasets. The main real-world testing dataset 
used throughout this repository are the TFL JamCam videos which are 
almost real-time videos provided by Transport for London (TfL) across 
100s of locations across london. They are useful for this project as 
they cover many different types of locations which helps validate the 
robustness of our proposed solution. This repository is provided 
as an installable python module as many command-line utilities are 
provided, most important of which is the microservice which the 
accompanying frontend and backend rely on. 
Refer to the [Web](https://github.com/MEng-Team-Project/MEng-Team-Project-Web)
repository for more information.

## Quick Start Guide

### Install the Python Package

You can install this python package from a local clone of the git repo by
doing the following:

```bash
# Clone and install this repository
git clone https://github.com/MEng-Team-Project/MEng-Team-Project-ML
python -m pip install -e MEng-Team-Project-ML/

# Get yolov8_tracking and yolov8 tensorrt submodule(s)
git submodule init
git submodule update

# Get yolov8 submodule for yolov8_tracking
cd yolov8_tracking
git submodule init
git submodule update
```

## Run the Microservice

To run the microservice, run the following code:

```bash
python -m traffic_ml.bin.microservice --analysis_dir "PATH_TO/MEng-Team-Project-Web/server/analysis"
```

## Source File Formats

- Accepts .mp4 source files. This is enforced during video stream
  upload by only accepting .mp4 files and live streams should be converted
  using either ffmpeg dynamically, or when yolov8 outputs a video it
  also needs ffmpeg to convert it to a lower bit-rate .mp4.
- Also accepts .m3u8 playlist files or .ts HLS livestream video segment files
  for analysing realtime IP video streams.

## Annotate a Video



## Notebooks

This section contains a detailed explanation for the contents and purpose
of each notebook.

<details><summary>1. JSON Attempt.ipynb</summary>

Analysis of the initial JSON files produced
in the original draft version of our proposed model. Notebook contains
code used to determine road routes, code used to calculate counts of
object types along routes, etc.

</details>

<details><summary>2. SQLite3 Attempt.ipynb</summary>

Changed recording of analytics from YOLOv7
and ClassySORT to use SQLite3 as the recorded format. This saved
information was extremely raw and ill conceived as it required
complex and difficult post-processing to get any kind of useful
information from.

</details>

<details><summary>3. SQLite3 StrongSORT.ipynb</summary>

Switched from YOLOv7 to YOLOv8 and
switched object tracking algorithm from SORT to StrongSORT which
gigantically improves performance. StrongSORT has a lower IDs
(identity switching) rate compared to SORT of 4470 compared to
1066, respectively on MOT20 [ref](https://github.com/dyhBUPT/StrongSORT).
This means that the SORT algorithm is identifying 4.19x more objects
than StrongSORT so it's association between detections and tracking
the same object across time is highly unstable.

</details>

<details><summary>4. Model Validation.ipynb</summary>

Contains a demonstration of how to validate
the predictions (pred) of our ML pipeline against ground truth (GT) annotations.
GT and pred data converted to MOT16 format and formally evaluated and compared
using `motmetrics` lib.

</details>

## HTTP API

This section contains each of the HTTP endpoints for the microservice,
with an explanation for each endpoint, along with it's expected and
optional parameters.

<details><summary>Initialise stream analysis</summary>

POST: `http://localhost:6000/api/init` 
```
   Body Parameters: 
   stream - Absolute video stream path
```

</details>

<details><summary>Retrieve existing high-level analytics</summary>

POST: `http://localhost:6000/api/analysis`
```
   Body Parameters: 
   stream  - Stream ID to get data for. 
   raw     - (Optional) Provide the raw detection information in the result 
   start   - (Optional) Start frame 
   end     - (Optional) End frame 
   classes - (Optional) List of COCO class labels to filter detections by 
   trk_fmt - (Optional) Either `first_last` or `entire`. This will either 
             include the first and last anchor points for an object in the 
             route, or it will include the entire route for the requested 
             portion of the video. By default, returns `first_last`
```

</details>

<details><summary>Retrieve existing low-level granular, route related analytics</summary>

POST: `http://localhost:6000/api/routes` 
```
   Body Parameters: 
   stream  - Stream ID to get data for. 
   regions - Route region polygon information 
   start   - (Optional) Start frame 
   end     - (Optional) End frame 
   classes - (Optional) List of COCO class labels to filter detections by
```

</details>

## TensorRT (YOLOv8 and StrongSORT)

### Overview

It is essential to export and test TensorRT using Linux.
It is possible to run TensorRT on Windows using the `.zip` package found
on the website, by the `tensorrt` module for Python is not really supported.

### YOLOv8

<details><summary>1. Export Engine</summary>

To export a model, use this command to export the PyTorch .pt model
to ONNX format .onnx:

```bash
python3 YOLOv8-TensorRT/export-det.py \
--weights yolov8l.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 32 3 640 640 \
--device cuda:0
```

Then you need to build a TensorRT engine with static settings which
will perform inference later. The execution device also needs to be
fixed here (GPU or CPU).

```bash
python3 YOLOv8-TensorRT/build.py \
--weights yolov8l.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--input-shape 32 3 640 640 \
--device cuda:0
```

</details>

<details><summary>2. Inference</summary>

To test run inference of the detection model and get timing profiling data,
run the following command:

```bash
python3 YOLOv8-TensorRT/infer-det-para.py \
--engine yolov8l.engine \
--source vid \
--batch-size 32
```

</details>

<details><summary>Profiling (Benchmark Performance)</summary>

To profile every single component of the TensorRT engine with an existing
model, run the following command:

```bash
python3 YOLOv8-TensorRT/trt-profile.py --engine yolov8s.engine --device cuda:0
```

</details>

### StrongSORT

This export requires version `8.0.20` of the `ultralytics` module.

```bash
pip install ultralytics==8.0.20
```

And then run the following command:

```bash
python3 yolov8_tracking/trackers/reid_export.py \
--device 0 \
--verbose \
--include engine \
--batch-size 32 \
--dynamic
```