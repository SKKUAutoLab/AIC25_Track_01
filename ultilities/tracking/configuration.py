#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
from __future__ import annotations

# ROOT_DATA_FOLDER = '/media/vsw/Data1/MTMC_Tracking_2025_500/'       # Data root
ROOT_DATA_FOLDER = '/media/vsw/Data1/MTMC_Tracking_2025/'       # Data root
DETECTION_RESULTS_FOLDER = 'ExtractFrames/image_result_test/'      # Folder contains results from detection phase
CROP_IMAGE_FOLDER = '/image_croped_test'                            # Cropped Image from Detection Results
FULL_IMAGE_FOLDER = 'ExtractFrames/images_extract_full/'            # Full size Image extract from videos


RESULTS_FOLDER = 'ExtractFrames/'
POSE_RESULTS = '/pose_results'
REID_RESULTS = '/reid_results'
MAX_FRAME = 9000

# FOR SINGLE VIEW TRACKING
SV_TRACK_RESULTS = 'track_sv_info'
SV_TRACK_VISUALIZATION = 'track_viz_allclass'
SV_VIZ_FLAG = True
SV_RESULT_FLAG = True

# FOR MULTIVIEW TRACKING
TOTAL_FRAME = 9000
FRAME_STEP  = 10
STATIC_DM_FOL  = 'static_dm_results'
DYNAMIC_DM_FOL = 'dynamic_dm_results'  # cai nay la chua thong tin cua single view tracking, da duoc modified cuar single view tracking

MV_TRACK_RESULTS = 'track_mv_info.txt'
MV_TRACK_VISUALIZATION = 'track_viz_allclass'
MV_VIZ_FLAG = True
