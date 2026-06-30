#!/bin/bash

# stop at the first error
set -e

conda create -n tracking python=3.10 -y
conda activate tracking

conda install -c conda-forge opencv
conda install -c conda-forge scipy
pip install open3d==0.19.0
conda install tqdm

pip install lap, cython_bbox, shapely

echo "Finish installation successfully"
