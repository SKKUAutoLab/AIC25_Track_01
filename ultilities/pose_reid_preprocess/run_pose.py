import os
import json
from tqdm import tqdm

import argparse
import glob
import os
from pathlib import Path
from typing import List
import shutil
from collections import defaultdict
import random

import cv2
import numpy as np
import torch
import re
from PIL import Image


import warnings
warnings.filterwarnings("ignore")
from mmengine.logging import MMLogger
MMLogger.get_instance('mmpose').setLevel('ERROR')

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
register_all_modules()

obj_keys = ['object type', 'object id', '3d location', '3d bounding box scale', '3d bounding box rotation', '2d bounding box visible']
calib_dict_key = ['type', 'id', 'coordinates', 'scaleFactor', 'translationToGlobalCoordinates', 'attributes', 'intrinsicMatrix', 'extrinsicMatrix', 'cameraMatrix', 'homography']


import ultils as io_utils

import configuration as config


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (np.integer, np.floating)):
			return obj.item()
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)


def run_pose_detection_per_scene(scene):
	config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
	checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
	model = init_model(config_file, checkpoint_file, device='cuda:0')

	# root_folder_test = '/media/vsw/Data1/MTMC_Tracking_2025_500/ExtractFrames/image_result_test'
	# pose_folder_test = '/media/vsw/Data1/MTMC_Tracking_2025_500/demo_500/'
	# 'aicity_test_crop_pose'

	cur_test_scene = config.ROOT_DATA_FOLDER + config.DETECTION_RESULTS_FOLDER + scene + config.CROP_IMAGE_FOLDER
	cur_pose_scene = config.ROOT_DATA_FOLDER + config.RESULTS_FOLDER + scene + config.POSE_RESULTS
	io_utils.generate_nested_folders(cur_pose_scene)

	print('Make list of cropped images...')
	image_files = [f for f in os.listdir(cur_test_scene) if os.path.isfile(os.path.join(cur_test_scene, f))]
	print('No of image files: ', len(image_files))

	for img_file in tqdm(image_files, desc=scene):
		class_id = img_file.split('_')[-1].split('.')[0]
		if class_id == '0':
			img = cv2.imread(cur_test_scene + '/' + img_file)
			h, w, _ = img.shape
			bbox_list = [[0,0,w,h]]
			results = inference_topdown(model, img, bboxes=bbox_list, bbox_format='xywh')

			p_result = results[0]
			if (len(p_result.pred_instances.keypoints) == 1):
				bbox = p_result.pred_instances.bboxes[0] #xyxy format
				kpc = p_result.pred_instances.keypoints[0]
				kps = p_result.pred_instances.keypoint_scores[0]
				kpcs = np.concatenate((kpc, kps[:, np.newaxis]), axis=1)
				kpcs[:, 0] -= bbox[0]
				kpcs[:, 1] -= bbox[1]
				score = np.mean(kpcs[:, 2:3])
				json_data = [
					{
						'keypoints': kpcs,
						'category_id': 1,
						'is_target': True,
						'bbox': [0, 0, bbox[2] - bbox[0], bbox[3]-bbox[1]],
						'score': score,
					}
				]
				json.dump(json_data, open(cur_pose_scene + '/' + img_file + '_keypoints.json', 'w'), indent=4, cls=NumpyEncoder)


def run_pose_detection():

	scene_list = [
		'Warehouse_017',
		'Warehouse_018',
		'Warehouse_019',
		'Warehouse_020',
	]

	from multiprocessing import Pool, set_start_method
	import multiprocessing as mp
	mp.set_start_method('spawn', force=True)
	number_of_thread = 4
	p = Pool(number_of_thread)
	p.map(run_pose_detection_per_scene, scene_list)


if __name__ == "__main__":
	# print(config.ROOT_DATA_FOLDER)
	run_pose_detection()

