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

# Get yolov8_tracking submodule
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

## TODO

- [x] TFL JamCam Video Scraper Datasets
  - [x] Create dataset for multiple conditions
     - [x] Day time conditions
           [Raw Videos](https://drive.google.com/drive/u/2/folders/1igKtgK_b13TBwwnDnX5_8y_3ij6R31i_),
           [Analysis Videos](https://drive.google.com/drive/u/2/folders/1TYaEDctAyxikJD2Oj717cl2KbSIDA2HB)
     - [x] Night time (incl. snow) conditions \
           [Raw Videos](https://drive.google.com/drive/u/2/folders/1lSoRB_HmgSLUxehpu7O44jQFB9U6hdWC),
           [Analysis Videos](https://drive.google.com/drive/u/2/folders/1JOJKVzakrFLt5tC5PpMq6zl6SGPwMopl)
- [ ] Custom Object Analysis Tested
   - [x] Bus (Detect Objects in Bus Lane)
      - Camera ID: 03752, contains a bus stop. This is used as the test set.
      - [Requirement](https://docs.google.com/document/d/1Q0TwboSrRgvXywVp9VgA9of2G4NjYQBYJ2YSKbkoS-o/edit#bookmark=id.hnc674tzl7ba)
   - [x] People (This is tested throughout the day time and night time datasets)
   - [ ] HGV
      - YOLOv8 not currently fine-tuned for HGVs yet.

## File Formats

- Accepts .mp4 source files. This is enforced during video stream
  upload by only accepting .mp4 files and live streams should be converted
  using either ffmpeg dynamically, or when yolov8 outputs a video it
  also needs ffmpeg to convert it to a lower bit-rate .mp4.
- Also accepts .m3u8 playlist files or .ts HLS livestream video segment files
  for analysing realtime IP video streams.

## Notebooks

- `1. JSON Attempt.ipynb.ipynb` Analysis of the initial JSON files produced
  in the original draft version of our proposed model. Notebook contains
  code used to determine road routes, code used to calculate counts of
  object types along routes, etc.
- `2. SQLite3 Attempt.ipynb` Changed recording of analytics from YOLOv7
  and ClassySORT to use SQLite3 as the recorded format. This saved
  information was extremely raw and ill conceived as it required
  complex and difficult post-processing to get any kind of useful
  information from.
- `3. SQLite3 StrongSORT.ipynb` Switched from YOLOv7 to YOLOv8 and
  switched object tracking algorithm from SORT to StrongSORT which
  gigantically improves performance. StrongSORT has a lower IDs
  (identity switching) rate compared to SORT of 4470 compared to
  1066, respectively on MOT20 [ref](https://github.com/dyhBUPT/StrongSORT).
  This means that the SORT algorithm is identifying 4.19x more objects
  than StrongSORT so it's association between detections and tracking
  the same object across time is highly unstable.