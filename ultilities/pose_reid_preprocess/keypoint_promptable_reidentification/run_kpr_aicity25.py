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

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import configuration as config

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
        kpcs = entry['keypoints']

        for kpc in kpcs:
            kpc[0] =  max(0, min(kpc[0], bbox[2] - 1))
            kpc[1] =  max(0, min(kpc[1], bbox[3] - 1))
        

def load_kpr_samples(image_files, image_folder_pth, pose_folder_pth):
    # Initialize an empty list to store the samples
    samples = []
    name_samples = []
    # Iterate over the image files and construct each sample dynamically
    for img_name in image_files:
        class_id = img_name.split('_')[-1].split('.')[0]
        if class_id == '0':
            # Construct full paths
            img_path = os.path.join(image_folder_pth, img_name)
            json_path = os.path.join(pose_folder_pth, img_name + '_keypoints.json')

            # Load the image
            img = cv2.imread(img_path)

            # Load the keypoints from the JSON file
            with open(json_path, 'r') as json_file:
                keypoints_data = json.load(json_file)
                clip_keypoints_data(keypoints_data)

            # Initialize lists to hold keypoints
            keypoints_xyc = []
            negative_kps = []

            # Process the keypoints data
            for entry in keypoints_data:
                if entry["is_target"]:
                    keypoints_xyc.append(entry["keypoints"])
                else:
                    negative_kps.append(entry["keypoints"])

            assert len(keypoints_xyc) == 1, "Only one target keypoint set is supported for now."

            # Convert lists to numpy arrays
            keypoints_xyc = np.array(keypoints_xyc[0])
            negative_kps = np.array(negative_kps)

            # Create the sample dictionary
            sample = {
                "image": img,
                "keypoints_xyc": keypoints_xyc,  # the positive prompts indicating the re-identification target
                "negative_kps": negative_kps,  # the negative keypoints indicating other pedestrians
            }
            # Append the sample to the list
            samples.append(sample)
            name_samples.append(img_name)
    return samples, name_samples


def run_kpr_a_scene(scene):

    kpr_cfg = build_config(config_path="configs/aicity_kpr/kpr_aicity_market_test.yaml")
    kpr_cfg.use_gpu = torch.cuda.is_available() # already done in build_config(...), but can be overwritten here
    extractor = KPRFeatureExtractor(kpr_cfg)

    cur_test_scene = config.ROOT_DATA_FOLDER + config.DETECTION_RESULTS_FOLDER + scene + config.CROP_IMAGE_FOLDER
    cur_pose_scene = config.ROOT_DATA_FOLDER + config.RESULTS_FOLDER + scene + config.POSE_RESULTS
    cur_reid_scene = config.ROOT_DATA_FOLDER + config.RESULTS_FOLDER + scene + config.REID_RESULTS

    generate_nested_folders(cur_reid_scene)

    print('Make list of cropped images...')
    image_files = [f for f in os.listdir(cur_test_scene) if os.path.isfile(os.path.join(cur_test_scene, f))]
    frameID_to_images = defaultdict(list)
    for img_file in tqdm(image_files, desc= scene + ' frameID searching'):
        fid = img_file.split('_')[5]  # extract frameID from filename
        frameID_to_images[fid].append(img_file)
    print('Total frames: ', len(frameID_to_images.keys()))

    for frameID in tqdm(frameID_to_images.keys(), desc= scene + ' extract ReID feats'):
        frame_dict = {}
        # Sort images by camID
        image_files_per_frame = frameID_to_images[frameID]
        camID_to_images = defaultdict(list)
        for img_file in image_files_per_frame:
            cid = img_file.split('_')[2]  # extract camID from filename
            camID_to_images[cid].append(img_file)

        for camID in camID_to_images.keys():
            img_list = camID_to_images[camID]
            samples, name_samples = load_kpr_samples(img_list, cur_test_scene, cur_pose_scene)
            if len(samples) > 0:
                _, embeddings, visibility_scores, _ = extractor(samples)
                np_embeddings = embeddings.cpu().detach().numpy()
                np_vis_scrores= visibility_scores.cpu().detach().numpy()

                frame_dict[camID] = {
                    'embeddings': np_embeddings,
                    'visibility_scores': np_vis_scrores,
                    'image_names': name_samples
                }
            else:
                frame_dict[camID] = {
                    'embeddings': None,
                    'visibility_scores': None,
                    'image_names': None
                }
        json.dump(frame_dict, open(cur_reid_scene + '/' + frameID + '.json', 'w'), indent=4, cls=NumpyEncoder)

    return


if __name__ == "__main__":
    run_kpr_a_scene(scene="Warehouse_017")
    run_kpr_a_scene(scene="Warehouse_018")
    run_kpr_a_scene(scene="Warehouse_019")
    run_kpr_a_scene(scene="Warehouse_020")

