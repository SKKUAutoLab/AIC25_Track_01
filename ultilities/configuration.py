#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
from __future__ import annotations
import os

FOLDER_ROOT         = "/media/vsw/Data1/"
FOLDER_DATA_VERSION = "MTMC_Tracking_2025"
FOLDER_DATASET_MAIN = os.path.join(FOLDER_ROOT, FOLDER_DATA_VERSION)

FOLDER_INPUT                       = os.path.join(FOLDER_DATASET_MAIN, "ExtractFrames/lookup_table/")
FOLDER_INPUT_LOOKUP_TABLE          = os.path.join(FOLDER_DATASET_MAIN, "ExtractFrames/lookup_table/")
FOLDER_INPUT_TEST                  = os.path.join(FOLDER_DATASET_MAIN, "test/")
FOLDER_INPUT_FULL_EXTRACTION_IMAGE = os.path.join(FOLDER_DATASET_MAIN, "ExtractFrames/images_extract_full/")

FOLDER_OUTPUT        = os.path.join(FOLDER_DATASET_MAIN, "ExtractFrames/lookup_table/")
FOLDER_VISUALIZATION = os.path.join(FOLDER_DATASET_MAIN, "ExtractFrames/image_result/")
FOLDER_PROCESSING    = os.path.join(FOLDER_DATASET_MAIN, "ExtractFrames/")

NUMBER_IMAGE_PER_CAMERA = 9000
LIST_SCENE              = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]