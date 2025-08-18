#!/bin/bash

# stop at the first error
set -e

conda create --name kpr python=3.10
conda activate kpr
conda install  pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

pip install -r requirements_kpr.txt

cd ultilities/pose_reid_preprocess/keypoint_promptable_reidentification
pip install -e .

echo "Finish installation successfully"
