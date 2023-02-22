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
"""Google Drive utility functions."""

import requests

def save_response_content(response, destination):
    """NOTE: Don't call this directly. Saves the Google Drive file if a valid response has been received."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_confirm_token(response):
    """NOTE: Don't call this directly. Gets an access token used to download the file."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def download_file_from_google_drive(id, destination):
    """Downloads a Google Drive file using its ID to a local destination.
    
    args:
        id          - Shareable link file ID
        destination - Download directory destination"""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_gdrive_id(url):
    """Gets the file ID needed to download a Google Drive file from
    a shareable URL."""
    try:
        url_parts = url.split("/")
        d_index   = url_parts.index("d")
        file_id   = url_parts[d_index + 1]
        return file_id
    except:
        return None

def run():
    TEST_FILE_LINK = "https://drive.google.com/file/d/1g_YASd9Rs_eAw4H_inGcFnDwQDmsj-fo/view?usp=share_link"
    destination = "./00001.01350_2022-12-07T15-35-24.000Z.mp4"
    gdrive_id      = get_gdrive_id(TEST_FILE_LINK)
    if gdrive_id:
        download_file_from_google_drive(gdrive_id, destination)
    else:
        print("Invalid shareable URL link:", TEST_FILE_LINK)

if __name__ == "__main__":
    run()