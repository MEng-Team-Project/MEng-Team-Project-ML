# MIT License
# 
# Copyright (c) 2022 MEng-Team-Project
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
"""Scrape target videos from TFL JamCams indefinitely to build a test dataset
for the traffic analysis library."""

import time

from absl import app
from absl import flags

from traffic_ml.lib.tfl import TFLScraper, get_targets, download_metadata

FLAGS = flags.FLAGS
flags.DEFINE_string("video_dir", None, "Directory to store scraped TFL JamCam videos")
flags.DEFINE_string("targets", "targets.txt", "Text file containing TFL JamCam ids to scrape")

flags.mark_flag_as_required("video_dir")

def go():
    targets_fname = FLAGS.targets
    video_dir     = FLAGS.video_dir
    targets       = get_targets(targets_fname)
    scraper       = TFLScraper(video_dir)

    # Download metadata
    download_metadata()

    # Download a video for each target
    for target in targets:
        scraper.download_video(target)
    
def main(unused_argv):
    while True:
        go()
        time.sleep(60)
        
def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)