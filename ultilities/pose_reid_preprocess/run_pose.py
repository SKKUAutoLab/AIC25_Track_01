import json
import multiprocessing as mp
import os

import cv2
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
from mmengine.logging import MMLogger
MMLogger.get_instance('mmpose').setLevel('ERROR')

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
register_all_modules()

import ultils as io_utils
import configuration as config


SCENES = ['Warehouse_017', 'Warehouse_018', 'Warehouse_019', 'Warehouse_020']
POSE_WORKER_COUNT = 4


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (np.integer, np.floating)):
			return obj.item()
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)


class ScenePoseEstimator:
	CONFIG_FILE = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
	CHECKPOINT_FILE = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

	def __init__(self, scene):
		self.scene = scene
		self.crop_folder = config.ROOT_DATA_FOLDER + config.DETECTION_RESULTS_FOLDER + scene + config.CROP_IMAGE_FOLDER
		self.pose_folder = config.ROOT_DATA_FOLDER + config.RESULTS_FOLDER + scene + config.POSE_RESULTS

	def run(self):
		model = init_model(self.CONFIG_FILE, self.CHECKPOINT_FILE, device='cuda:0')
		io_utils.generate_nested_folders(self.pose_folder)
		for img_file in tqdm(self._list_image_files(), desc=self.scene):
			if not self._is_person_crop(img_file):
				continue
			self._estimate_and_save(model, img_file)

	def _list_image_files(self):
		print('Make list of cropped images...')
		image_files = [f for f in os.listdir(self.crop_folder) if os.path.isfile(os.path.join(self.crop_folder, f))]
		print('No of image files: ', len(image_files))
		return image_files

	@staticmethod
	def _is_person_crop(img_file):
		return img_file.split('_')[-1].split('.')[0] == '0'

	def _estimate_and_save(self, model, img_file):
		img = cv2.imread(self.crop_folder + '/' + img_file)
		h, w, _ = img.shape
		results = inference_topdown(model, img, bboxes=[[0, 0, w, h]], bbox_format='xywh')
		p_result = results[0]
		if len(p_result.pred_instances.keypoints) != 1:
			return
		json_data = self._build_keypoints(p_result)
		json.dump(json_data, open(self.pose_folder + '/' + img_file + '_keypoints.json', 'w'), indent=4, cls=NumpyEncoder)

	@staticmethod
	def _build_keypoints(p_result):
		bbox = p_result.pred_instances.bboxes[0]
		kpc = p_result.pred_instances.keypoints[0]
		kps = p_result.pred_instances.keypoint_scores[0]
		kpcs = np.concatenate((kpc, kps[:, np.newaxis]), axis=1)
		kpcs[:, 0] -= bbox[0]
		kpcs[:, 1] -= bbox[1]
		score = np.mean(kpcs[:, 2:3])
		return [
			{
				'keypoints': kpcs,
				'category_id': 1,
				'is_target': True,
				'bbox': [0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]],
				'score': score,
			}
		]


def run_pose_detection_per_scene(scene):
	ScenePoseEstimator(scene).run()


def run_pose_detection():
	mp.set_start_method('spawn', force=True)
	pool = mp.Pool(POSE_WORKER_COUNT)
	pool.map(run_pose_detection_per_scene, SCENES)


if __name__ == "__main__":
	run_pose_detection()
