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
git clone https://github.com/MEng-Team-Project/MEng-Team-Project-ML
python -m pip install -e MEng-Team-Project-ML/
git submodule init
git submodule update
```

## Run the Microservice

To run the microservice, run the following code:

```bash
python -m traffic_ml.bin.microservice --analysis_dir "./yolov7-segmentation/runs/predict-seg/"
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
   - [ ] HGV
   - [x] People (This is tested throughout the day time and night time datasets)

## File Formats

- Accepts .mp4 source files. This is enforced during video stream
  upload by only accepting .mp4 files and live streams should be converted
  using either ffmpeg dynamically, or when yolov7 outputs a video it
  also needs ffmpeg to convert it to a lower bit-rate .mp4.
- Also accepts .m3u8 playlist files or .ts HLS livestream video segment files
  for analysing realtime IP video streams.

## Utility Files

- `traffic_ml/bin/check` is designed to verify the analytical information being
  produced by the forked `yolov7-segmentation` submodule is producing
  sensible data. If you run the script (after setting the correct
  parameters within the file), you should find that the `Tracked Detections`
  is the lowest, followed by the `Segments` and then the `Routes` is the
  highest. The reason for this is because there are more routes per frame
  than there are bounding box detections. Then there are more segments
  than full tracked detections as (I believe) there can be multiple segments
  per individual object. Then the tracked detections are complete objects
  which are being tracked by the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter).
  Run this using:
  ```bash
  python -m traffic_ml.bin.check --analysis_dir "" --fname ""
  ```
<!--
- `live_metadata.py` displays the multimedia playlist which informs
  clients which .ts files (livestream video segments) to download in which order
  to correctly view the livestream. Useful for understanding how the .ts files
  should be fed into the microservice for batch processing the videos for
  analysis when we come to deploy the system for real, and also providing a
  user who is viewing our client with real-time route tracking / object recognition.
-->

## YoloV7 Outputs

To move all of yolov7's predicted outputs from it's individual experimental
directories into one single directory, you can use the following command to
do this (assumes bash like terminal, you can use MinGW or WSL on Windows).

```bash
cp .../predict-seg/**/*.mp4 .../destination
```

## Notebooks

- `main.ipynb` Analysis of the initial JSON files produced in the original
  draft version of our proposed model. Notebook contains code used to determine
  road routes, code used to calculate counts of object types along routes, etc.