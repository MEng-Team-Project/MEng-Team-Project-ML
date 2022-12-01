# Traffic-Core

## About

Microservice designed to accept analysis batch / live jobs and information
requests from a backend.

## Assumptions

- Only accepts .mp4 source files. This is enforced during video stream
  upload by only accepting .mp4 files and live streams should be converted
  using either ffmpeg dynamically, or when yolov7 outputs a video it
  also needs ffmpeg to convert it to a lower bit-rate .mp4.

## Utility Files

- `check.py` is designed to verify the analytical information being
  produced by the forked `yolov7-segmentation` submodule is producing
  sensible data. If you run the script (after setting the correct
  parameters within the file), you should find that the `Tracked Detections`
  is the lowest, followed by the `Segments` and then the `Routes` is the
  highest. The reason for this is because there are more routes per frame
  than there are bounding box detections. Then there are more segments
  than full tracked detections as (I believe) there can be multiple segments
  per individual object. Then the tracked detections are complete objects
  which are being tracked by the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter).