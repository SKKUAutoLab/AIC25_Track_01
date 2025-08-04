#!/bin/bash

# stop at the first error
set -e

conda env create -f pose_env.yml
conda activate mmpose
python run_pose.py

echo "Finish installation successfully"
