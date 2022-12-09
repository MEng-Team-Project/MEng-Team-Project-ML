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
"""Module setuptools script."""
from setuptools import setup

description = """Traffic-ML - Traffic Analysis Deep Learning Library

Traffic-ML is the Python component of the MEng-Team-Project for the
Semantic identification of moving vehicles through traffic junctions to infer
relevant usage of each route. This Python module identifies, classifies and counts
vehicles on a particular junction.
Read the README at https://github.com/MEng-Team-Project/MEng-Team-Project-ML
for more information.
"""


setup(
    name='traffic-ml',
    version='1.0.0',
    description='Traffic Analysis Deep Learning Library',
    long_description=description,
    long_description_content_type="text/markdown",
    author=['Tada Makepeace', 'Ben Garaffa'],
    author_email=['up904749@gmail.com'],
    license='MIT License',
    keywords=[
        'traffic analysis',
        'traffic',
        'semantic identification',
        'route analysis',
        'tensorrt',
        'production'
    ],
    url='https://github.com/MiscellaneousStuff/tlol-py',
    packages=[
        'traffic_ml',
        'traffic_ml.bin',
    ],
    install_requires=[
        'absl-py',
        'requests'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)