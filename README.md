# MEng-Team-Project-ML

## About

Microservice designed to accept analysis batch / live jobs and information
requests from a backend.

## TODO

- [ ] TFL JamCam Video Scraper Datasets
  - [ ] Create dataset for multiple conditions
     - [ ] Day time conditions
     - [x] Night time (incl. snow) conditions \
           [Raw Videos](https://drive.google.com/drive/u/2/folders/1lSoRB_HmgSLUxehpu7O44jQFB9U6hdWC),
           [Analysis Videos](https://drive.google.com/drive/u/2/folders/1JOJKVzakrFLt5tC5PpMq6zl6SGPwMopl)

## File Formats

- Accepts .mp4 source files. This is enforced during video stream
  upload by only accepting .mp4 files and live streams should be converted
  using either ffmpeg dynamically, or when yolov7 outputs a video it
  also needs ffmpeg to convert it to a lower bit-rate .mp4.
- Also accepts .m3u8 playlist files or .ts HLS livestream video segment files
  for analysing realtime IP video streams.

## Utility Files (Old)

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
- `live_metadata.py` displays the multimedia playlist which informs
  clients which .ts files (livestream video segments) to download in which order
  to correctly view the livestream. Useful for understanding how the .ts files
  should be fed into the microservice for batch processing the videos for
  analysis when we come to deploy the system for real, and also providing a
  user who is viewing our client with real-time route tracking / object recognition.

# YoloV7 Outputs

To move all of yolov7's predicted outputs from it's individual experimental
directories into one single directory, you can use the following command to
do this (assumes bash like terminal, you can use WSL on Windows).

```bash
cp .../predict-seg/**/*.mp4 .../destination
```