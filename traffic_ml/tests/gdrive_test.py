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
"""Google Drive utilities testing."""

import ffmpeg
import os

from absl.testing import absltest

from traffic_ml.tests import utils
from traffic_ml.lib.gdrive import get_gdrive_id, download_file_from_google_drive

TEST_FILE_LINK        = "https://drive.google.com/file/d/1g_YASd9Rs_eAw4H_inGcFnDwQDmsj-fo/view?usp=share_link"
TEST_FILE_DESTINATION = "./00001.01350_2022-12-07T15-35-24.000Z.mp4"
TARGET_DURATION_TS    = 133632

class TestGDriveDownload(utils.TestCase):
    def test_gdrive_download(self):
        gdrive_id = get_gdrive_id(TEST_FILE_LINK)
        if gdrive_id:
            download_file_from_google_drive(gdrive_id, TEST_FILE_DESTINATION)
            metadata = ffmpeg.probe(TEST_FILE_DESTINATION)["streams"][0]
            self.assertEqual(metadata["duration_ts"], TARGET_DURATION_TS)
        else:
            print("Invalid shareable URL link:", TEST_FILE_LINK)

    def tearDown(self):
        super(TestGDriveDownload, self).tearDown()

        os.remove(TEST_FILE_DESTINATION)

if __name__ == "__main__":
    absltest.main()