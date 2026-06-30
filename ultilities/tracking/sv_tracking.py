from botsort.bot_sort import BoTSORT

import cv2
import os
import numpy as np
import argparse
import json
from tqdm import tqdm

import pita_utils as tracking_utils

import configuration as config


dataset_root = config.ROOT_DATA_FOLDER
sub_folder_tracking_results = config.RESULTS_FOLDER
max_frame_idx = 9000
imgs_fol = config.FULL_IMAGE_FOLDER
pose_fol = config.POSE_RESULTS
reid_fol = config.REID_RESULTS

track_viz_fol = config.SV_TRACK_VISUALIZATION
track_sv_info_fol = config.SV_TRACK_RESULTS
viz_flag = config.SV_VIZ_FLAG
save_track_info_flag = config.SV_RESULT_FLAG

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 165, 255),
    (128, 0, 128),
    (255, 255, 0)
]

SCENES = ['Warehouse_017', 'Warehouse_018', 'Warehouse_019', 'Warehouse_020']


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SingleViewTracker:
    OTHER_CLASS_IDS = (2, 3, 4, 5)

    def __init__(self, args):
        self.args = args

    def run(self, scene_name):
        print('Run tracking on scene: ' + scene_name)
        camera_keys = tracking_utils.get_camera_keys(dataset_root=dataset_root, imgs_fol=imgs_fol, scene_name=scene_name)
        print('Camkey: ', camera_keys)
        other_classes_json = self._load_other_classes(scene_name)
        sv_save_list = self._prepare_output_folders(scene_name, camera_keys)
        person_trackers, class_trackers = self._build_trackers(camera_keys)
        for frame_idx in tqdm(range(max_frame_idx), desc=scene_name):
            self._process_frame(scene_name, camera_keys, frame_idx, other_classes_json, person_trackers, class_trackers, sv_save_list)

    @staticmethod
    def _load_other_classes(scene_name):
        other_classes_json_pth = dataset_root + config.DETECTION_RESULTS_FOLDER + scene_name + '/' + scene_name + '_json_cropped_test_non-person.json'
        with open(other_classes_json_pth, 'r') as json_file:
            return json.load(json_file)

    @staticmethod
    def _prepare_output_folders(scene_name, camera_keys):
        track_viz_fol_full = dataset_root + sub_folder_tracking_results + scene_name + '/' + track_viz_fol
        if not os.path.exists(track_viz_fol_full):
            os.makedirs(track_viz_fol_full)
        track_sv_info_fol_full = dataset_root + sub_folder_tracking_results + scene_name + '/' + track_sv_info_fol
        if not os.path.exists(track_sv_info_fol_full):
            os.makedirs(track_sv_info_fol_full)
        sv_save_list = []
        for camkey in camera_keys:
            viz_cam_pth = track_viz_fol_full + '/Camera_' + camkey
            sv_save_pth = track_sv_info_fol_full + '/Camera_' + camkey
            if not os.path.exists(viz_cam_pth):
                os.makedirs(viz_cam_pth)
            if not os.path.exists(sv_save_pth):
                os.makedirs(sv_save_pth)
            sv_save_list.append(sv_save_pth)
        return sv_save_list

    def _build_trackers(self, camera_keys):
        person_trackers = []
        class_trackers = []
        for _ in camera_keys:
            person_trackers.append(BoTSORT(args=self.args))
            class_trackers.append({cls_id: BoTSORT(args=self.args) for cls_id in self.OTHER_CLASS_IDS})
        return person_trackers, class_trackers

    def _process_frame(self, scene_name, camera_keys, frame_idx, other_classes_json, person_trackers, class_trackers, sv_save_list):
        cam_viz_list = self._load_viz_images(scene_name, camera_keys, frame_idx)
        save_trackInfo_dict = {camkey: [] for camkey in camera_keys}
        person_reid_infos = self._load_person_reid(scene_name, frame_idx)
        for camIdx in range(len(camera_keys)):
            self._track_persons(scene_name, camera_keys, camIdx, person_reid_infos, person_trackers, cam_viz_list, save_trackInfo_dict)
        other_classes_info = other_classes_json[str(frame_idx)]
        for camIdx in range(len(camera_keys)):
            self._track_other_classes(camera_keys, camIdx, other_classes_info, class_trackers, cam_viz_list, save_trackInfo_dict)
        self._save_frame(camera_keys, frame_idx, save_trackInfo_dict, sv_save_list)

    @staticmethod
    def _load_viz_images(scene_name, camera_keys, frame_idx):
        cam_viz_list = []
        if viz_flag:
            for camkey in camera_keys:
                cam_img_pth = dataset_root + imgs_fol + scene_name + '/Camera_' + camkey + '/' + str(frame_idx).zfill(8) + '.jpg'
                cam_viz_list.append(cv2.imread(cam_img_pth))
        return cam_viz_list

    @staticmethod
    def _load_person_reid(scene_name, frame_idx):
        reid_info_pth = dataset_root + sub_folder_tracking_results + scene_name + reid_fol + '/' + str(frame_idx).zfill(7) + '.json'
        with open(reid_info_pth, 'r') as json_file:
            return json.load(json_file)

    def _track_persons(self, scene_name, camera_keys, camIdx, person_reid_infos, person_trackers, cam_viz_list, save_trackInfo_dict):
        camkey = camera_keys[camIdx]
        reid_feat_person, reid_viss_person = self._person_reid_features(camkey, person_reid_infos)
        det_persons, pose_persons = self._person_detections(scene_name, camkey, person_reid_infos)

        det_person_arr = np.array(det_persons, dtype=np.float32)
        pose_person_arr = np.array(pose_persons, dtype=np.float32)
        if det_person_arr.shape[0] > 0:
            reid_feat_person_arr = np.array(reid_feat_person, dtype=np.float32)
            reid_viss_person_arr = np.array(reid_viss_person, dtype=np.float32)
            reid_viss_person_expand_arr = reid_viss_person_arr[:, :, np.newaxis]
            reid_person_arr = np.concatenate((reid_feat_person_arr, reid_viss_person_expand_arr), axis=2)
        else:
            reid_person_arr = []

        overlaps_list = tracking_utils.get_overlapping_indices(det_person_arr, iou_thresh=0.85)
        det_person_arr = np.delete(det_person_arr, overlaps_list, axis=0)
        reid_person_arr = np.delete(reid_person_arr, overlaps_list, axis=0)
        pose_person_arr = np.delete(pose_person_arr, overlaps_list, axis=0)
        online_targets = person_trackers[camIdx].update(det_person_arr, reid_person_arr, pose_person_arr)

        if save_track_info_flag:
            for t in online_targets:
                save_trackInfo_dict[camkey].append({
                    'id': t.track_id,
                    'tlwh': t.tlwh,
                    'pose': t.pose,
                    'feat': t.curr_feat,
                    'viss': t.curr_viss,
                    'obj_cls': 0,
                })
        if viz_flag:
            for t in online_targets:
                cam_viz_list[camIdx] = self._draw_person(cam_viz_list[camIdx], t)

    @staticmethod
    def _person_reid_features(camkey, person_reid_infos):
        if camkey in person_reid_infos.keys():
            return person_reid_infos[camkey]['embeddings'], person_reid_infos[camkey]['visibility_scores']
        return [], []

    @staticmethod
    def _person_detections(scene_name, camkey, person_reid_infos):
        det_persons = []
        pose_persons = []
        if camkey not in person_reid_infos.keys():
            return det_persons, pose_persons
        if person_reid_infos[camkey]['image_names'] is None:
            return det_persons, pose_persons
        detected_person_infos = person_reid_infos[camkey]['image_names']
        for person_idx in range(len(detected_person_infos)):
            a_person = detected_person_infos[person_idx]
            a_person_info = a_person.split('_')
            detect_x1 = float(a_person_info[6])
            detect_y1 = float(a_person_info[7])
            detect_x2 = float(a_person_info[8])
            detect_y2 = float(a_person_info[9])
            detect_cf = float(a_person_info[10].split('.j')[0])
            detect_person_input = [person_idx, detect_x1, detect_y1, detect_x2, detect_y2, detect_cf, 0]
            det_persons.append(detect_person_input)
            a_person_pose_pth = dataset_root + sub_folder_tracking_results + scene_name + pose_fol + '/' + a_person + '_keypoints.json'
            pose_persons.append(tracking_utils.get_pose_infor_from_path(a_person_pose_pth, detect_person_input))
        return det_persons, pose_persons

    @staticmethod
    def _draw_person(cam_img, t):
        tlwh = t.tlwh
        tl_p = (int(tlwh[0]), int(tlwh[1]))
        br_p = (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
        cv2.rectangle(cam_img, tl_p, br_p, colors[0], 1, cv2.LINE_AA)
        cv2.putText(cam_img, str(t.track_id), tl_p, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[0], thickness=2)
        return tracking_utils.visualize_pose(cam_img, t.pose, pcolor=colors[0], kp_thresh=0.5)

    def _track_other_classes(self, camera_keys, camIdx, other_classes_info, class_trackers, cam_viz_list, save_trackInfo_dict):
        camkey = camera_keys[camIdx]
        if camkey not in other_classes_info.keys():
            return
        per_cam_other_classes_info = np.array(other_classes_info[camkey], dtype=np.float32)
        for obj_cls in self.OTHER_CLASS_IDS:
            per_class_info = per_cam_other_classes_info[per_cam_other_classes_info[:, -1] == obj_cls]
            non_person_online_targets = class_trackers[camIdx][obj_cls].update_no_reid(per_class_info)
            if viz_flag:
                for t in non_person_online_targets:
                    self._draw_other(cam_viz_list[camIdx], t)
            if save_track_info_flag:
                for t in non_person_online_targets:
                    save_trackInfo_dict[camkey].append({
                        'id': t.track_id,
                        'tlwh': t.tlwh,
                        'pose': None,
                        'feat': None,
                        'viss': None,
                        'obj_cls': t.obj_cls,
                    })

    @staticmethod
    def _draw_other(cam_img, t):
        tlwh = t.tlwh
        tl_p = (int(tlwh[0]), int(tlwh[1]))
        br_p = (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
        cv2.rectangle(cam_img, tl_p, br_p, colors[int(t.obj_cls)], 1, cv2.LINE_AA)
        cv2.putText(cam_img, str(t.track_id), tl_p, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[int(t.obj_cls)], thickness=2)

    @staticmethod
    def _save_frame(camera_keys, frame_idx, save_trackInfo_dict, sv_save_list):
        if not save_track_info_flag:
            return
        for camIdx in range(len(camera_keys)):
            json.dump(save_trackInfo_dict[camera_keys[camIdx]],
                      open(sv_save_list[camIdx] + '/' + str(frame_idx).zfill(7) + '.json', 'w'),
                      indent=4, cls=NumpyEncoder)


def make_parser_singleview():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="test mot20.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.2, help='threshold for rejecting low appearance similarity reid matches')
    opt = parser.parse_args()
    return opt


def main_singleview():
    opt = make_parser_singleview()
    tracker = SingleViewTracker(opt)
    for scene_name in SCENES:
        tracker.run(scene_name)


if __name__ == '__main__':
    main_singleview()
