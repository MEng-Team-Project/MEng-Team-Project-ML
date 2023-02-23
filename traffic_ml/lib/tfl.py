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

import os
import requests
import json

from lxml import etree
from dotenv import dotenv_values

def download_metadata(METADATA_URL="https://api.tfl.gov.uk/Place/Type/JamCam"):
    response = requests.get(METADATA_URL)
    open("metadata.json", "wb").write(response.content)

def download_camera_updates(
    CAMERA_UPDATES_XML,
    CAMERA_UPDATES_URL="https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/"):
    response = requests.get(CAMERA_UPDATES_URL)
    open(CAMERA_UPDATES_XML, "wb").write(response.content)

def convert_camera_updates(xml_fname):
    camera_updates = {}
    with open(xml_fname, "rb") as f:
        content = f.read()
        root = etree.fromstring(content)
        for elem in root:
            if "Contents" in elem.tag:
                key = None
                last_modified = None
                for param in elem:
                    if "Key" in param.tag:
                        if param.text.endswith(".mp4"):
                            key = param.text[:-4]
                    elif "LastModified" in param.tag:
                        last_modified = param.text
                camera_updates[key] = last_modified
    return camera_updates

def get_camera_updates(CAMERA_UPDATES_XML="camera_updates.xml", force_update=False):
    try:
        if not os.path.exists(CAMERA_UPDATES_XML) or \
           os.path.exists(CAMERA_UPDATES_XML) and force_update:
            download_camera_updates(CAMERA_UPDATES_XML)
        contents = convert_camera_updates(CAMERA_UPDATES_XML)
        return contents
    except Exception as e:
        print("Error retrieving camera updates:", e)
        return None

def get_metadata():
    if not os.path.exists("./metadata.json"):
        download_metadata()
    with open("./metadata.json") as f:
        content = json.loads(f.read())
        return content

def get_targets(fname="targets.txt"):
    with open(fname) as f:
        return f.read().split("\n")


class TFLScraper(object):
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.base_url  = "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/"
        self.metadata  = self.get_metadata(get_metadata())
        self.updates   = get_camera_updates(force_update=True)

    def download_video(self, id):
        update = self.updated(id)
        print(id, update, self.metadata[id])
        if update:
            url = f"{self.base_url}{id}.mp4"
            response = requests.get(url)
            update = update.replace(":", "-")
            dst = os.path.join(self.video_dir, f"{id}_{update}.mp4")
            open(dst, "wb").write(response.content)

    def get_metadata(self, metadata):
        out = {}
        for m in metadata:
            out[m["id"].split("_")[1]] = {
                "name": m["commonName"],
                "lat":  m["lat"],
                "long": m["lon"]
            }
        return out

    def updated(self, target_id):
        updates = self.updates
        if not target_id in updates.keys():
            return False

        update  = updates[target_id]
        update  = update.replace(":", "-")

        videos  = os.listdir(self.video_dir)
        videos  = [fi for fi in videos
                   if target_id in fi]
        video_dates = [".".join(fi.split("_")[1].split(".")[0:-1])
                       for fi in videos]
        if not update in video_dates:
            return update
        else:
            return False