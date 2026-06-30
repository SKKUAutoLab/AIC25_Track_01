import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm

import pita_utils as my_utils

import configuration as config

from mv_tracking_lib.multiview_tracking import MTrack


max_frame_idx = config.MAX_FRAME
dataset_root = config.ROOT_DATA_FOLDER
imgs_fol = config.FULL_IMAGE_FOLDER

sub_folder_tracking_results = config.RESULTS_FOLDER
dynamic_dm_fol = config.DYNAMIC_DM_FOL

mv_track_viz_fol = config.MV_TRACK_VISUALIZATION
mv_track_file_name = config.MV_TRACK_RESULTS

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 165, 255),
    (128, 0, 128),
    (255, 255, 0)
]

object_type_name = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
}

SCENES = ['Warehouse_017', 'Warehouse_018', 'Warehouse_019', 'Warehouse_020']


class MultiViewTracker:
    def __init__(self, args):
        self.args = args

    def run(self, scene_name):
        print('Run mv_tracking on scene: ' + scene_name)
        camera_keys = my_utils.get_camera_keys(dataset_root=dataset_root, imgs_fol=imgs_fol, scene_name=scene_name)
        print('Camkey: ', camera_keys)
        calib_info = my_utils.load_calib_infos(dataset_root=dataset_root, scene_name=scene_name)

        save_results_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + mv_track_file_name
        print(f"{save_results_pth=}")

        save_viz_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + mv_track_viz_fol
        my_utils.generate_nested_folders(save_viz_pth)

        bev_dm_list = self._bev_paths(scene_name, camera_keys)
        bev_img = cv2.imread(dataset_root + '/test/' + scene_name + '/map.png')

        mv_tracker = MTrack(camera_keys, create_new_objs=(scene_name == 'Warehouse_020'))
        scene_id = scene_name.split('_')[1].split('-')[0]
        cam_calib = calib_info[camera_keys[0]]

        with open(save_results_pth, "w") as result_f:
            for frame_idx in tqdm(range(max_frame_idx), desc=scene_name):
                all_dets, all_poses, all_feats, all_visses = self._load_frame_objects(camera_keys, bev_dm_list, frame_idx)
                all_dets, all_poses, all_feats, all_visses = self._filter_detections(scene_name, all_dets, all_poses, all_feats, all_visses)
                mv_tracker.update(all_dets, all_poses, all_feats, all_visses, frame_idx)
                self._write_frame_results(result_f, scene_id, frame_idx, mv_tracker)
                self._render_bev_visualization(mv_tracker, bev_img, cam_calib)

    @staticmethod
    def _bev_paths(scene_name, camera_keys):
        bev_dm_list = []
        for i in range(len(camera_keys)):
            bev_dm_list.append(dataset_root + sub_folder_tracking_results + scene_name + '/' + dynamic_dm_fol + '/Camera_' + camera_keys[i])
        return bev_dm_list

    @staticmethod
    def _load_frame_objects(camera_keys, bev_dm_list, frame_idx):
        all_dets = []
        all_poses = []
        all_feats = []
        all_visses = []
        for camIdx in range(len(camera_keys)):
            cam_sv_info_pth = bev_dm_list[camIdx] + '/' + str(frame_idx).zfill(7) + '.json'
            with open(cam_sv_info_pth, 'r') as json_file:
                cam_sv_info_dict = json.load(json_file)
            if cam_sv_info_dict['det_info'] is None:
                continue
            if len(all_dets) == 0:
                all_dets = np.array(cam_sv_info_dict['det_info'])
                all_poses = np.array(cam_sv_info_dict['poses'])
                all_feats = np.array(cam_sv_info_dict['feats'])
                all_visses = np.array(cam_sv_info_dict['visses'])
            else:
                all_dets = np.concatenate((all_dets, cam_sv_info_dict['det_info']), axis=0)
                all_poses = np.concatenate((all_poses, cam_sv_info_dict['poses']), axis=0)
                all_feats = np.concatenate((all_feats, cam_sv_info_dict['feats']), axis=0)
                all_visses = np.concatenate((all_visses, cam_sv_info_dict['visses']), axis=0)
        return all_dets, all_poses, all_feats, all_visses

    @staticmethod
    def _filter_detections(scene_name, all_dets, all_poses, all_feats, all_visses):
        keep_det_indices = my_utils.remove_dets_based_on_dm_ratio(all_dets, dm_ratio_thresh=0.05)
        all_dets = all_dets[np.where(keep_det_indices)[0]]
        all_poses = all_poses[np.where(keep_det_indices)[0]]
        all_feats = all_feats[np.where(keep_det_indices)[0]]
        all_visses = all_visses[np.where(keep_det_indices)[0]]
        if scene_name == 'Warehouse_018-enhance':
            keep_det_indices = my_utils.remove_too_tight_dets(all_dets, [0, 4, 5], w_threshold=30)
            all_dets = all_dets[np.where(keep_det_indices)[0]]
            all_poses = all_poses[np.where(keep_det_indices)[0]]
            all_feats = all_feats[np.where(keep_det_indices)[0]]
            all_visses = all_visses[np.where(keep_det_indices)[0]]
        return all_dets, all_poses, all_feats, all_visses

    def _write_frame_results(self, result_f, scene_id, frame_idx, mv_tracker):
        for cur_tracked_obj in mv_tracker.tracked_objs:
            self._write_result_line(result_f, scene_id, cur_tracked_obj.obj_cls, cur_tracked_obj.global_id, frame_idx, cur_tracked_obj.bev_coor, self._max_height(cur_tracked_obj.hist_heights))
        for cur_missing_obj in mv_tracker.missing_objs:
            height = self._max_height(cur_missing_obj.hist_heights)
            if len(cur_missing_obj.temp_bev_coor_during_missing):
                self._write_result_line(result_f, scene_id, cur_missing_obj.obj_cls, cur_missing_obj.global_id, frame_idx, cur_missing_obj.temp_bev_coor_during_missing[-1], height)
        for cur_tracked_obj in mv_tracker.tracked_others:
            self._write_result_line(result_f, scene_id, cur_tracked_obj.obj_cls, cur_tracked_obj.global_id, frame_idx, cur_tracked_obj.bev_coor, -1.0)
        for cur_missing_obj in mv_tracker.missing_others:
            if len(cur_missing_obj.temp_bev_coor_during_missing):
                self._write_result_line(result_f, scene_id, cur_missing_obj.obj_cls, cur_missing_obj.global_id, frame_idx, cur_missing_obj.temp_bev_coor_during_missing[-1], -1.0)

    @staticmethod
    def _max_height(hist_heights):
        if len(hist_heights):
            return np.max(hist_heights)
        return -1.0

    @staticmethod
    def _write_result_line(result_f, scene_id, obj_cls, global_id, frame_idx, bev_coor, height):
        result_f.write(str(int(scene_id)) + " "
            + str(int(obj_cls)) + " "
            + str(int(global_id)) + " "
            + str(int(frame_idx)) + " "
            + str((bev_coor[0])) + " "
            + str((bev_coor[1])) + " "
            + str((bev_coor[2])) + " "
            + "-1.0 "
            + "-1.0 "
            + str(height) + " "
            + "-1.0 \n")

    def _render_bev_visualization(self, mv_tracker, bev_img, cam_calib):
        frame_bev = bev_img.copy()
        for cur_tracked_obj in mv_tracker.tracked_objs:
            bev_center = self._bev_center(cur_tracked_obj.bev_coor, cam_calib)
            cv2.putText(frame_bev, str(cur_tracked_obj.global_id), (int(bev_center[0]), 1080 - int(bev_center[1])),
                cv2.FONT_HERSHEY_PLAIN, 2, colors[0], 2, cv2.LINE_AA)
        for missing_obj in mv_tracker.missing_objs:
            bev_coor = missing_obj.last_bev_coor
            if len(missing_obj.temp_bev_coor_during_missing):
                bev_coor = missing_obj.temp_bev_coor_during_missing[-1]
            bev_center = self._bev_center(bev_coor, cam_calib)
            cv2.putText(frame_bev, str(missing_obj.global_id), (int(bev_center[0]), 1080 - int(bev_center[1])),
                cv2.FONT_ITALIC, 2, colors[0], 2, cv2.LINE_AA)
        for cur_tracked_obj in mv_tracker.tracked_others:
            bev_center = self._bev_center(cur_tracked_obj.bev_coor, cam_calib)
            cv2.putText(frame_bev, str(cur_tracked_obj.global_id), (int(bev_center[0]), 1080 - int(bev_center[1])),
                cv2.FONT_HERSHEY_PLAIN, 2, colors[int(cur_tracked_obj.obj_cls)], 2, cv2.LINE_AA)
        for cur_missing_other in mv_tracker.missing_others:
            bev_coor = cur_missing_other.last_bev_coor
            if len(cur_missing_other.temp_bev_coor_during_missing):
                bev_coor = cur_missing_other.temp_bev_coor_during_missing[-1]
            bev_center = self._bev_center(bev_coor, cam_calib)
            cv2.putText(frame_bev, str(cur_missing_other.global_id), (int(bev_center[0]), 1080 - int(bev_center[1])),
                cv2.FONT_ITALIC, 2, colors[int(cur_missing_other.obj_cls)], 2, cv2.LINE_AA)
        cv2.putText(frame_bev, object_type_name[0] + " " + str(len(mv_tracker.tracked_objs))
                                + " - " + str(len(mv_tracker.missing_objs)),
                                    (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, colors[0], 2, cv2.LINE_AA)
        for idx in range(1, 6, 1):
            tracked_obj_list = mv_tracker.return_others_by_class_id(idx, True)
            missing_obj_list = mv_tracker.return_others_by_class_id(idx, False)
            cv2.putText(frame_bev, object_type_name[idx] + " " + str(len(tracked_obj_list))
                                + " - " + str(len(missing_obj_list)),
                                    (100, 100 + 70 * idx), cv2.FONT_HERSHEY_PLAIN, 3, colors[idx], 2, cv2.LINE_AA)
        return frame_bev

    @staticmethod
    def _bev_center(bev_coor, cam_calib):
        bev_center = [bev_coor[0], bev_coor[1]]
        bev_center[0] += cam_calib['translationToGlobalCoordinates']['x']
        bev_center[1] += cam_calib['translationToGlobalCoordinates']['y']
        bev_center *= np.array(cam_calib['scaleFactor'])
        return bev_center


def make_parser_multiview():
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
    parser.add_argument('--create_new_person', type=bool, default=False)
    opt = parser.parse_args()
    return opt


def main_multiview():
    opt = make_parser_multiview()
    tracker = MultiViewTracker(opt)
    for scene_name in SCENES:
        tracker.run(scene_name)


if __name__ == '__main__':
    main_multiview()
