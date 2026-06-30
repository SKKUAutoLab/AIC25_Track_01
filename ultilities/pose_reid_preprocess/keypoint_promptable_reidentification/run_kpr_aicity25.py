import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import configuration as config


SCENES = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_nested_folders(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


def clip_keypoints_data(keypoints_data):
    for entry in keypoints_data:
        bbox = entry['bbox']
        for kpc in entry['keypoints']:
            kpc[0] = max(0, min(kpc[0], bbox[2] - 1))
            kpc[1] = max(0, min(kpc[1], bbox[3] - 1))


def load_kpr_samples(image_files, image_folder_pth, pose_folder_pth):
    samples = []
    name_samples = []
    for img_name in image_files:
        if img_name.split('_')[-1].split('.')[0] != '0':
            continue
        samples.append(_build_sample(img_name, image_folder_pth, pose_folder_pth))
        name_samples.append(img_name)
    return samples, name_samples


def _build_sample(img_name, image_folder_pth, pose_folder_pth):
    img = cv2.imread(os.path.join(image_folder_pth, img_name))
    with open(os.path.join(pose_folder_pth, img_name + '_keypoints.json'), 'r') as json_file:
        keypoints_data = json.load(json_file)
        clip_keypoints_data(keypoints_data)
    keypoints_xyc = []
    negative_kps = []
    for entry in keypoints_data:
        if entry["is_target"]:
            keypoints_xyc.append(entry["keypoints"])
        else:
            negative_kps.append(entry["keypoints"])
    assert len(keypoints_xyc) == 1, "Only one target keypoint set is supported for now."
    return {
        "image": img,
        "keypoints_xyc": np.array(keypoints_xyc[0]),
        "negative_kps": np.array(negative_kps),
    }


class SceneReidExtractor:
    def __init__(self, scene):
        self.scene = scene
        self.crop_folder = config.ROOT_DATA_FOLDER + config.DETECTION_RESULTS_FOLDER + scene + config.CROP_IMAGE_FOLDER
        self.pose_folder = config.ROOT_DATA_FOLDER + config.RESULTS_FOLDER + scene + config.POSE_RESULTS
        self.reid_folder = config.ROOT_DATA_FOLDER + config.RESULTS_FOLDER + scene + config.REID_RESULTS

    def run(self):
        extractor = self._build_extractor()
        generate_nested_folders(self.reid_folder)
        frames = self._group_images_by_frame()
        print('Total frames: ', len(frames.keys()))
        for frameID in tqdm(frames.keys(), desc=self.scene + ' extract ReID feats'):
            frame_dict = self._extract_frame(extractor, frames[frameID])
            json.dump(frame_dict, open(self.reid_folder + '/' + frameID + '.json', 'w'), indent=4, cls=NumpyEncoder)

    @staticmethod
    def _build_extractor():
        kpr_cfg = build_config(config_path="configs/aicity_kpr/kpr_aicity_market_test.yaml")
        kpr_cfg.use_gpu = torch.cuda.is_available()
        return KPRFeatureExtractor(kpr_cfg)

    def _group_images_by_frame(self):
        print('Make list of cropped images...')
        image_files = [f for f in os.listdir(self.crop_folder) if os.path.isfile(os.path.join(self.crop_folder, f))]
        frameID_to_images = defaultdict(list)
        for img_file in tqdm(image_files, desc=self.scene + ' frameID searching'):
            frameID_to_images[img_file.split('_')[5]].append(img_file)
        return frameID_to_images

    def _extract_frame(self, extractor, image_files_per_frame):
        camID_to_images = defaultdict(list)
        for img_file in image_files_per_frame:
            camID_to_images[img_file.split('_')[2]].append(img_file)
        frame_dict = {}
        for camID in camID_to_images.keys():
            frame_dict[camID] = self._extract_camera(extractor, camID_to_images[camID])
        return frame_dict

    def _extract_camera(self, extractor, img_list):
        samples, name_samples = load_kpr_samples(img_list, self.crop_folder, self.pose_folder)
        if len(samples) == 0:
            return {'embeddings': None, 'visibility_scores': None, 'image_names': None}
        _, embeddings, visibility_scores, _ = extractor(samples)
        return {
            'embeddings': embeddings.cpu().detach().numpy(),
            'visibility_scores': visibility_scores.cpu().detach().numpy(),
            'image_names': name_samples,
        }


if __name__ == "__main__":
    for scene in SCENES:
        SceneReidExtractor(scene).run()
