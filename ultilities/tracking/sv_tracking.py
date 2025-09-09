from botsort.bot_sort import BoTSORT

import cv2
import os
import time
import numpy as np
import argparse
import json
from collections import defaultdict
from tqdm import tqdm

import pita_utils as tracking_utils

import configuration as config

# from tracking.mv_tracking.

# Mutual
dataset_root = config.ROOT_DATA_FOLDER
sub_folder_tracking_results = config.RESULTS_FOLDER
max_frame_idx = 9000 # Default 9000
imgs_fol = config.FULL_IMAGE_FOLDER
pose_fol = config.POSE_RESULTS
reid_fol = config.REID_RESULTS

# Single view
track_viz_fol = config.SV_TRACK_VISUALIZATION
track_sv_info_fol = config.SV_TRACK_RESULTS
viz_flag = config.SV_VIZ_FLAG   # co can luu hinh anh visualization xuong khong
save_track_info_flag = config.SV_RESULT_FLAG  # cos luu ket qua single view tracking xuong khong

colors = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 165, 255),    # Orange
    (128, 0, 128),    # Purple
    (255, 255, 0)     # Cyan
]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    

def run_tracking_per_scene(scene_name, args):
    print('Run tracking on scene: ' + scene_name)
    camera_keys = tracking_utils.get_camera_keys(dataset_root=dataset_root, imgs_fol=imgs_fol, scene_name=scene_name)
    print('Camkey: ', camera_keys)

    novacarter_botsort_trackers = []
    transporter_botsort_trackers = []
    fourierGR1T2_botsort_trackers = []
    agilitidigit_botsort_trackers = []
    botsort_trackers = [] # For person only
    viz_fol_list = []
    sv_save_list = []

    # Load detection infor for other classes - except Person
    other_classes_json_pth = dataset_root + config.DETECTION_RESULTS_FOLDER + scene_name + '/' + scene_name + '_json_cropped_test_non-person.json'
    with open(other_classes_json_pth, 'r') as json_file:
        other_classes_json = json.load(json_file)
    # print(other_classes_json)

    track_viz_fol_full = dataset_root + sub_folder_tracking_results + scene_name + '/' + track_viz_fol
    if not os.path.exists(track_viz_fol_full):
        os.makedirs(track_viz_fol_full)
    track_sv_info_fol_full = dataset_root + sub_folder_tracking_results + scene_name + '/' + track_sv_info_fol
    if not os.path.exists(track_sv_info_fol_full):
        os.makedirs(track_sv_info_fol_full)

    for i in range(len(camera_keys)):
        botsort_trackers.append(BoTSORT(args=args))
        novacarter_botsort_trackers.append(BoTSORT(args=args))
        transporter_botsort_trackers.append(BoTSORT(args=args))
        fourierGR1T2_botsort_trackers.append(BoTSORT(args=args))
        agilitidigit_botsort_trackers.append(BoTSORT(args=args))

        viz_cam_pth = track_viz_fol_full + '/Camera_' + camera_keys[i]
        sv_save_pth = track_sv_info_fol_full + '/Camera_' + camera_keys[i]

        if not os.path.exists(viz_cam_pth):
            os.makedirs(viz_cam_pth)
        if not os.path.exists(sv_save_pth):
            os.makedirs(sv_save_pth)
        viz_fol_list.append(viz_cam_pth)
        sv_save_list.append(sv_save_pth)

    for frame_idx in tqdm(range(max_frame_idx), desc=scene_name):
        # For viz
        cam_viz_list = []
        if viz_flag:
            for camIdx in range(0, len(camera_keys)):
                camkey = camera_keys[camIdx]
                cam_img_pth = dataset_root + imgs_fol + scene_name + '/Camera_' + camkey + '/' + str(frame_idx).zfill(8) + '.jpg'
                cam_img = cv2.imread(cam_img_pth)
                cam_viz_list.append(cam_img)

        save_trackInfo_dict = {}
        for camIdx in range(0, len(camera_keys)):
            camkey = camera_keys[camIdx]
            save_trackInfo_dict[camkey] = []

        # For person class
        reid_info_pth = dataset_root + sub_folder_tracking_results + scene_name + reid_fol + '/' + str(frame_idx).zfill(7) + '.json'

        with open(reid_info_pth, 'r') as json_file:
            person_reid_infos = json.load(json_file) # camkey -> ('embeddings', 'visibility_scores', 'image_names')
        
        for camIdx in range(0, len(camera_keys)):
            camkey = camera_keys[camIdx]
            det_persons = []
            pose_persons = []
            if camkey in person_reid_infos.keys():
                reid_feat_person = person_reid_infos[camkey]['embeddings']
                reid_viss_person = person_reid_infos[camkey]['visibility_scores']
            else:
                reid_feat_person = []
                reid_viss_person = []

            # Load and prepare detection and pose results
            if camkey in person_reid_infos.keys():
                if person_reid_infos[camkey]['image_names'] is not None:
                    detected_person_infos = person_reid_infos[camkey]['image_names']
                    no_person = len(detected_person_infos)
                    for person_idx in range(0, no_person):
                        a_person = detected_person_infos[person_idx]
                        a_person_info = a_person.split('_')
                        detect_idx = float(a_person_info[0])
                        detect_x1 = float(a_person_info[6])
                        detect_y1 = float(a_person_info[7])
                        detect_x2 = float(a_person_info[8])
                        detect_y2 = float(a_person_info[9])
                        detect_cf = float(a_person_info[10].split('.j')[0])
                        detect_class = 0 # Person
                        detect_person_input = [person_idx, detect_x1, detect_y1, detect_x2, detect_y2, detect_cf, detect_class]
                        det_persons.append(detect_person_input)

                        # dataset_root + 'demo_500/' + scene_name + '/' + reid_fol 
                        a_person_pose_pth = dataset_root + sub_folder_tracking_results + scene_name + pose_fol + '/' + a_person + '_keypoints.json'
                        pose_person_input = tracking_utils.get_pose_infor_from_path(a_person_pose_pth, detect_person_input)
                        pose_persons.append(pose_person_input)
            
            det_person_arr = np.array(det_persons, dtype=np.float32)
            pose_person_arr = np.array(pose_persons, dtype=np.float32)
            if det_person_arr.shape[0] > 0:
                reid_feat_person_arr = np.array(reid_feat_person, dtype=np.float32)
                reid_viss_person_arr = np.array(reid_viss_person, dtype=np.float32)
                reid_viss_person_expand_arr = reid_viss_person_arr[:, :, np.newaxis]
                reid_person_arr = np.concatenate((reid_feat_person_arr, reid_viss_person_expand_arr), axis=2)
            else:
                reid_person_arr = []

            # Update person to their trackers
            overlaps_list = tracking_utils.get_overlapping_indices(det_person_arr, iou_thresh=0.85)
            # if (len(overlaps_list) > 0):
            #     print(det_person_arr[overlaps_list])
            det_person_arr = np.delete(det_person_arr, overlaps_list, axis=0)
            reid_person_arr = np.delete(reid_person_arr, overlaps_list, axis=0)
            pose_person_arr = np.delete(pose_person_arr, overlaps_list, axis=0)
            online_targets = botsort_trackers[camIdx].update(det_person_arr, reid_person_arr, pose_person_arr)

            if save_track_info_flag:
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    tpose = t.pose
                    tfeat = t.curr_feat
                    tviss = t.curr_viss
                    # tfeat_hist = list(t.features)
                    # tviss_hist = list(t.visses)
                    save_trackInfo_dict[camkey].append(
                        {
                            'id': tid,
                            'tlwh': tlwh,
                            'pose': tpose,
                            'feat': tfeat,
                            'viss': tviss,
                            # 'feat_hist': tfeat_hist,
                            # 'viss_hist': tviss_hist,
                            'obj_cls': 0,
                        }
                    )
                    
            if viz_flag:
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    tpose = t.pose
                    tl_p = (int(tlwh[0]), int(tlwh[1]))
                    br_p = (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
                    cv2.rectangle(cam_viz_list[camIdx], tl_p, br_p, colors[0], 1, cv2.LINE_AA)
                    cv2.putText(cam_viz_list[camIdx], str(tid), tl_p, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[0], thickness=2)
                    cam_viz_list[camIdx] = tracking_utils.visualize_pose(cam_viz_list[camIdx], tpose, pcolor=colors[0],kp_thresh=0.5)

        # For other classes
        other_classes_info = other_classes_json[str(frame_idx)]
        for camIdx in range(0, len(camera_keys)):
            camkey = camera_keys[camIdx]
            per_cam_other_classes_info = []
            if camkey in other_classes_info.keys():
                per_cam_other_classes_info = np.array(other_classes_info[camkey], dtype=np.float32)

                for obj_cls in range(2,6,1):
                    # person_idx, detect_x1, detect_x2, detect_y1, detect_y2, detect_cf, detect_class
                    per_class_info = per_cam_other_classes_info[per_cam_other_classes_info[:, -1] == obj_cls]

                    if obj_cls == 2:
                        non_person_online_targets = novacarter_botsort_trackers[camIdx].update_no_reid(per_class_info)    
                    elif obj_cls == 3:
                        non_person_online_targets = transporter_botsort_trackers[camIdx].update_no_reid(per_class_info)    
                    elif obj_cls == 4:
                        non_person_online_targets = fourierGR1T2_botsort_trackers[camIdx].update_no_reid(per_class_info) 
                    else:
                        non_person_online_targets = agilitidigit_botsort_trackers[camIdx].update_no_reid(per_class_info)    

                    if viz_flag:
                        for t in non_person_online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            tclass = t.obj_cls
                            tl_p = (int(tlwh[0]), int(tlwh[1]))
                            br_p = (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
                            cv2.rectangle(cam_viz_list[camIdx], tl_p, br_p, colors[int(tclass)], 1, cv2.LINE_AA)
                            cv2.putText(cam_viz_list[camIdx], str(tid), tl_p, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[int(tclass)], thickness=2)

                    if save_track_info_flag:
                        for t in non_person_online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            tclass = t.obj_cls
                            save_trackInfo_dict[camkey].append(
                                {
                                    'id': tid,
                                    'tlwh': tlwh,
                                    'pose': None,
                                    'feat': None,
                                    'viss': None,
                                    'obj_cls': tclass
                                }
                            )

        if viz_flag:
            for camIdx in range(0, len(camera_keys)):    
                cv2.imwrite(viz_fol_list[camIdx] + '/' + str(frame_idx).zfill(7) + '.jpg', cam_viz_list[camIdx])
        
        if save_track_info_flag:
            for camIdx in range(0, len(camera_keys)):  
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
    run_tracking_per_scene('Warehouse_017', opt)
    run_tracking_per_scene('Warehouse_018', opt)
    run_tracking_per_scene('Warehouse_019', opt)
    run_tracking_per_scene('Warehouse_020', opt)


if __name__ == '__main__':
    main_singleview()