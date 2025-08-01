#!/bin/bash

# stop at the first error
set -e

# Create environment
conda create --name aic25_track1 python=3.10 -y
conda activate aic25_track1

# Install the required packages.
pip install poetry==1.2.0  # Python dependency management and packaging made easy.
pip install loguru==0.7.3  # Python logging made (stupidly) simple
pip install opencv-python==4.11.0.86  # OpenCV is a library of programming functions mainly aimed at real-time computer vision.
pip install easydict==1.13  # A lightweight dictionary for Python.
pip install tqdm==4.67.1  # A fast, extensible progress bar for Python and CLI.
pip install shapely==2.1.0  # A Python package for manipulation and analysis of planar geometric objects.
#pip install multipledispatch==1.0.0  # A generic function dispatcher in Python.
pip install multimethod==2.0 # A generic function dispatcher in Python.
pip install torchmetrics==1.7.1  # PyTorch native Metrics
pip install PyYAML==6.0.2  # PyYAML is a YAML parser and emitter for Python.
pip install h5py==3.14.0  # HDF5 is a file format and set of tools for managing complex data.
pip install open3d==0.19.0 # An open-source library that supports rapid development of software that deals with 3D data.
pip install scipy==1.15.3  # SciPy is a Python library used for scientific and technical computing.

# Run line by line
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install independent Ultralytics
cd third_party/ultralytics-8.3.141
pip install -e .

# In stall the project in editable mode.
rm -rf poetry.lock
poetry install --extras "dev"
rm -rf poetry.lock

echo "Finish installation successfully"
