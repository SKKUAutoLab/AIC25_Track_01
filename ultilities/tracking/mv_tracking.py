import cv2
import os
import time
import numpy as np
import argparse
import json
from collections import defaultdict, Counter
from tqdm import tqdm

import pita_utils as my_utils
import h5py

import configuration as config

from mv_tracking_lib.multiview_tracking import MTrack

# Mutual
max_frame_idx = config.MAX_FRAME
dataset_root = config.ROOT_DATA_FOLDER
imgs_fol = config.FULL_IMAGE_FOLDER

sub_folder_tracking_results = config.RESULTS_FOLDER
pose_fol = config.POSE_RESULTS
reid_fol = config.REID_RESULTS
track_viz_fol = config.SV_TRACK_VISUALIZATION
track_sv_info_fol = config.SV_TRACK_RESULTS
static_dm_fol = config.STATIC_DM_FOL
dynamic_dm_fol = config.DYNAMIC_DM_FOL # default


mv_track_viz_fol = config.MV_TRACK_VISUALIZATION
mv_track_file_name = config.MV_TRACK_RESULTS
mv_viz_flag = config.MV_VIZ_FLAG

colors = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 165, 255),    # Orange
    (128, 0, 128),    # Purple
    (255, 255, 0)     # Cyan
]

object_type_name = {
 0 : "Person", # red
 1 : "Forklift", # green
 2 : "NovaCarter", # blue
 3 : "Transporter", # Orange long, yellow car
 4 : "FourierGR1T2", # purple
 5 : "AgilityDigit", # Cyan
}

cam_colors = [
    (0, 0, 255),     # Red      0
    (0, 255, 0),     # Green    1
    (255, 0, 0),     # Blue     2
    (0, 255, 255),   # Yellow   3
    (255, 255, 0),   # Cyan     4
    (255, 0, 255),   # Magenta  5
    (0, 165, 255),   # Orange   6
    (128, 0, 128),   # Purple   7
    (128, 128, 0),   # Teal     8
    (0, 255, 191)    # Lime     9
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_calib_infos(scene_name):
    # with open(dataset_root + '/' + calib_fol + '/' + scene_name + '.json', 'r') as json_file:
    with open(dataset_root + '/test/' + scene_name + 'calibration.json', 'r') as json_file:
        scene_calib_info = json.load(json_file)
    
    calib_dict = {}
    sensors_calib_info = scene_calib_info['sensors']
    
    for item in sensors_calib_info:
        if item['type'] == 'camera':
            camkey = item['id'].split('_')[-1]
            calib_dict[camkey] = {
                'coordinates': item['coordinates'],
                'scaleFactor': item['scaleFactor'],
                'translationToGlobalCoordinates': item['translationToGlobalCoordinates'],
                'intrinsicMatrix': item['intrinsicMatrix'],
                'extrinsicMatrix': item['extrinsicMatrix'],
                'cameraMatrix': item['cameraMatrix'],
                'homography': item['homography'],
            }
    return calib_dict

    
def load_results_from_singview(result_pth, camkeys, frameIdx):

    dets_arr = []
    pose_arr = []
    feat_arr = []
    viss_arr = []

    fake_pose = np.zeros((17,3))
    fake_feat = np.zeros((6, 512))
    fake_viss = np.zeros((6))

    for idx in range(len(camkeys)):
        camkey = camkeys[idx]
        with open(result_pth[idx] + '/' + str(frameIdx).zfill(7) + '.json', 'r') as json_file:
            sv_result_idx = json.load(json_file)
            for obj in sv_result_idx:
                tid  = obj['id']
                tlwh = obj['tlwh']
                pose = obj['pose']
                feat = obj['feat']
                viss = obj['viss']
                tclass = obj['obj_cls']
                if not tclass is None:
                    dets_arr.append([idx, int(tclass), int(tid), 
                        int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])])
                    if pose is None or feat is None or viss is None:
                        pose_arr.append(fake_pose)
                        feat_arr.append(fake_feat)
                        viss_arr.append(fake_viss)
                    else:
                        pose_arr.append(pose)
                        feat_arr.append(feat)
                        viss_arr.append(viss)
    return np.array(dets_arr), np.array(pose_arr), np.array(feat_arr), np.array(viss_arr)

# cam_idx class_idx sv_track_idx x y w h ps_flag

def load_results_from_singview_1cam(result_pth, camkeys, camkey, frameIdx):

    dets_arr = []
    pose_arr = []
    feat_arr = []
    viss_arr = []

    fake_pose = np.zeros((17,3))
    fake_feat = np.zeros((6, 512))
    fake_viss = np.zeros((6))

    for idx in range(len(camkeys)):
        cur_camkey = camkeys[idx]
        if cur_camkey == camkey:
            with open(result_pth[idx] + '/' + str(frameIdx).zfill(7) + '.json', 'r') as json_file:
                sv_result_idx = json.load(json_file)
                for obj in sv_result_idx:
                    tid  = obj['id']
                    tlwh = obj['tlwh']
                    pose = obj['pose']
                    feat = obj['feat']
                    viss = obj['viss']
                    tclass = obj['obj_cls']
                    vis_pose_flag = 1
                    if not tclass is None:
                        if pose is None or feat is None or viss is None:
                            pose_arr.append(fake_pose)
                            feat_arr.append(fake_feat)
                            viss_arr.append(fake_viss)
                            vis_pose_flag = 0
                        else:
                            pose_arr.append(pose)
                            feat_arr.append(feat)
                            viss_arr.append(viss)
                        
                        dets_arr.append([idx, int(tclass), int(tid), 
                            int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]), vis_pose_flag])

    return np.array(dets_arr), np.array(pose_arr), np.array(feat_arr), np.array(viss_arr)


def run_mv_tracking(scene_name, args):
    print('Run mv_tracking on scene: ' + scene_name)
    camera_keys = my_utils.get_camera_keys(dataset_root=dataset_root, imgs_fol=imgs_fol, scene_name=scene_name)
    print('Camkey: ', camera_keys)

    calib_info = my_utils.load_calib_infos(dataset_root=dataset_root, scene_name=scene_name)

    # NOTE: luu ket qua cuoi cung cua multiview tracking
    # MARK:
    # SUGAR:
    save_results_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + mv_track_file_name
    result_f = open(save_results_pth, "w")

    # DEBUG:
    print(f"{save_results_pth=}")

    save_viz_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + mv_track_viz_fol
    my_utils.generate_nested_folders(save_viz_pth)

    sv_result_list = []
    # bev_sv_list = []
    bev_dm_list = []

    # mv_track_viz_pth = dataset_root + '/' + mv_track_viz_fol + '/' + scene_name
    # if not os.path.exists(mv_track_viz_pth):
    #     os.makedirs(mv_track_viz_pth)

    for i in range(len(camera_keys)):
        # For inputs
        sv_result_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + track_sv_info_fol + '/Camera_' + camera_keys[i]
        sv_result_list.append(sv_result_pth)
        bev_dm_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + dynamic_dm_fol + '/Camera_' + camera_keys[i]
        bev_dm_list.append(bev_dm_pth)
    
    # Lay cho map.png
    bev_map_pth = dataset_root + '/test/' + scene_name + '/map.png'
    bev_img = cv2.imread(bev_map_pth)

    create_new_obj_flag = False
    if scene_name == 'Warehouse_020':
        create_new_obj_flag = True

    mv_tracker = MTrack(camera_keys, create_new_objs=create_new_obj_flag)

    for frame_idx in tqdm(range(max_frame_idx), desc=scene_name):
        cam_calib = calib_info[camera_keys[0]] # Use cam 0 as default
        frame_bev = bev_img.copy()

        all_dets = []   # cam_idx class_idx sv_track_idx x y w h ps_flag x y z bev_flag
        all_poses = []
        all_feats = []
        all_visses = []

        cam_imgs = []

        # Load obj from single view track
        for camIdx in range(len(camera_keys)):
            cam_sv_info_pth = bev_dm_list[camIdx] + '/' + str(frame_idx).zfill(7) + '.json'
            with open(cam_sv_info_pth, 'r') as json_file:
                cam_sv_info_dict = json.load(json_file)
                if not cam_sv_info_dict['det_info'] is None:
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

        keep_det_indices = my_utils.remove_dets_based_on_dm_ratio(all_dets, dm_ratio_thresh=0.05)
        all_dets = all_dets[np.where(keep_det_indices)[0]]
        all_poses = all_poses[np.where(keep_det_indices)[0]]
        all_feats = all_feats[np.where(keep_det_indices)[0]]
        all_visses = all_visses[np.where(keep_det_indices)[0]]

        # FIXME: check here
        # scene 18 la co van de, nen nhung object nao nho qua thi bo qua
        if scene_name == 'Warehouse_018-enhance':
            keep_det_indices = my_utils.remove_too_tight_dets(all_dets, [0, 4, 5], w_threshold=30)
            all_dets = all_dets[np.where(keep_det_indices)[0]]
            all_poses = all_poses[np.where(keep_det_indices)[0]]
            all_feats = all_feats[np.where(keep_det_indices)[0]]
            all_visses = all_visses[np.where(keep_det_indices)[0]]

        r_other_dets = mv_tracker.update(all_dets, all_poses, all_feats, all_visses, frame_idx)

        # Write results
        # for person only
        scene_id = scene_name.split('_')[1].split('-')[0]
        for cur_tracked_obj in mv_tracker.tracked_objs:
            cur_t_h = -1.0
            if len(cur_tracked_obj.hist_heights):
                cur_t_h = np.max(cur_tracked_obj.hist_heights)
            result_f.write(str(int(scene_id)) + " " 
                + str(int(cur_tracked_obj.obj_cls)) + " "
                + str(int(cur_tracked_obj.global_id)) + " "
                + str(int(frame_idx)) + " "
                + str((cur_tracked_obj.bev_coor[0])) + " "
                + str((cur_tracked_obj.bev_coor[1])) + " "
                + str((cur_tracked_obj.bev_coor[2])) + " "
                + "-1.0 " # w
                + "-1.0 " # l
                + str(cur_t_h) + " " # h
                + "-1.0 \n" # yaw
                )

        # for missing person
        for cur_missing_obj in mv_tracker.missing_objs:
            cur_t_h = -1.0
            if len(cur_missing_obj.hist_heights):
                cur_t_h = np.max(cur_missing_obj.hist_heights)
            if len(cur_missing_obj.temp_bev_coor_during_missing):
                result_f.write(str(int(scene_id)) + " " 
                    + str(int(cur_missing_obj.obj_cls)) + " "
                    + str(int(cur_missing_obj.global_id)) + " "
                    + str(int(frame_idx)) + " "
                    + str((cur_missing_obj.temp_bev_coor_during_missing[-1][0])) + " "
                    + str((cur_missing_obj.temp_bev_coor_during_missing[-1][1])) + " "
                    + str((cur_missing_obj.temp_bev_coor_during_missing[-1][2])) + " "
                    + "-1.0 " # w
                    + "-1.0 " # l
                    + str(cur_t_h) + " " # h
                    + "-1.0 \n" # yaw
                    )
        # for other objects
        for cur_tracked_obj in mv_tracker.tracked_others:
            result_f.write(str(int(scene_id)) + " " 
                + str(int(cur_tracked_obj.obj_cls)) + " "
                + str(int(cur_tracked_obj.global_id)) + " "
                + str(int(frame_idx)) + " "
                + str((cur_tracked_obj.bev_coor[0])) + " "
                + str((cur_tracked_obj.bev_coor[1])) + " "
                + str((cur_tracked_obj.bev_coor[2])) + " "
                + "-1.0 " # w
                + "-1.0 " # h
                + "-1.0 " # l
                + "-1.0 \n" # yaw
                )

        # for missing other objects
        for cur_missing_obj in mv_tracker.missing_others:
            if len(cur_missing_obj.temp_bev_coor_during_missing):
                result_f.write(str(int(scene_id)) + " " 
                    + str(int(cur_missing_obj.obj_cls)) + " "
                    + str(int(cur_missing_obj.global_id)) + " "
                    + str(int(frame_idx)) + " "
                    + str((cur_missing_obj.temp_bev_coor_during_missing[-1][0])) + " "
                    + str((cur_missing_obj.temp_bev_coor_during_missing[-1][1])) + " "
                    + str((cur_missing_obj.temp_bev_coor_during_missing[-1][2])) + " "
                    + "-1.0 " # w
                    + "-1.0 " # h
                    + "-1.0 " # l
                    + "-1.0 \n" # yaw
                    )

        # visulization only
        cur_tracked_objs = mv_tracker.tracked_objs
        for cur_tracked_obj in cur_tracked_objs:
            bev_coor = cur_tracked_obj.bev_coor
            g_id = cur_tracked_obj.global_id
            bev_center = [bev_coor[0], bev_coor[1]]
            bev_center[0] +=  cam_calib['translationToGlobalCoordinates']['x']
            bev_center[1] +=  cam_calib['translationToGlobalCoordinates']['y']
            bev_center *= np.array(cam_calib['scaleFactor'])
            cv2.putText(frame_bev, str(g_id), (int(bev_center[0]), 1080 - int(bev_center[1])), 
                cv2.FONT_HERSHEY_PLAIN, 2, colors[0], 2, cv2.LINE_AA)

        # visulization only missing
        missing_objs = mv_tracker.missing_objs
        for missing_obj in missing_objs:
            bev_coor = missing_obj.last_bev_coor
            if len(missing_obj.temp_bev_coor_during_missing):
                bev_coor = missing_obj.temp_bev_coor_during_missing[-1]
            g_id = missing_obj.global_id
            bev_center = [bev_coor[0], bev_coor[1]]
            bev_center[0] +=  cam_calib['translationToGlobalCoordinates']['x']
            bev_center[1] +=  cam_calib['translationToGlobalCoordinates']['y']
            bev_center *= np.array(cam_calib['scaleFactor'])
            cv2.putText(frame_bev, str(g_id), (int(bev_center[0]), 1080 - int(bev_center[1])), 
                cv2.FONT_ITALIC, 2, colors[0], 2, cv2.LINE_AA)

        # visulization others
        cur_tracked_others = mv_tracker.tracked_others
        for cur_tracked_obj in cur_tracked_others:
            bev_coor = cur_tracked_obj.bev_coor
            g_id = cur_tracked_obj.global_id
            cur_tracked_obj_class = int(cur_tracked_obj.obj_cls)
            bev_center = [bev_coor[0], bev_coor[1]]
            bev_center[0] +=  cam_calib['translationToGlobalCoordinates']['x']
            bev_center[1] +=  cam_calib['translationToGlobalCoordinates']['y']
            bev_center *= np.array(cam_calib['scaleFactor'])
            cv2.putText(frame_bev, str(g_id), (int(bev_center[0]), 1080 - int(bev_center[1])),
                        cv2.FONT_HERSHEY_PLAIN, 2, colors[cur_tracked_obj_class], 2, cv2.LINE_AA)

        # visulization missing others
        cur_missing_others = mv_tracker.missing_others
        for cur_missing_other in cur_missing_others:
            bev_coor = cur_missing_other.last_bev_coor
            if len(cur_missing_other.temp_bev_coor_during_missing):
                bev_coor = cur_missing_other.temp_bev_coor_during_missing[-1]
            g_id = cur_missing_other.global_id
            cur_missing_other_class = int(cur_missing_other.obj_cls)
            bev_center = [bev_coor[0], bev_coor[1]]
            bev_center[0] +=  cam_calib['translationToGlobalCoordinates']['x']
            bev_center[1] +=  cam_calib['translationToGlobalCoordinates']['y']
            bev_center *= np.array(cam_calib['scaleFactor'])
            cv2.putText(frame_bev, str(g_id), (int(bev_center[0]), 1080 - int(bev_center[1])),
                        cv2.FONT_ITALIC, 2, colors[cur_missing_other_class], 2, cv2.LINE_AA)

        cv2.putText(frame_bev, object_type_name[0] + " " + str(len(mv_tracker.tracked_objs)) 
                                + " - " + str(len(mv_tracker.missing_objs)), 
                                    (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, colors[0], 2, cv2.LINE_AA)

        # visulization on BEV map (map.png)
        for idx in range(1,6,1):
            tracked_obj_list = mv_tracker.return_others_by_class_id(idx, True)
            missing_obj_list = mv_tracker.return_others_by_class_id(idx, False)
            cv2.putText(frame_bev, object_type_name[idx] + " " + str(len(tracked_obj_list)) 
                                + " - " + str(len(missing_obj_list)), 
                                    (100, 100 + 70 * idx), cv2.FONT_HERSHEY_PLAIN, 3, colors[idx], 2, cv2.LINE_AA)
        
        # cv2.namedWindow("bev_map", cv2.WINDOW_NORMAL)
        # cv2.imshow('bev_map', frame_bev)

        # imwrite, save on the disk
        # if mv_viz_flag:
        #     cv2.imwrite(save_viz_pth + '/' + str(frame_idx).zfill(7) + '.jpg', frame_bev)
        # cv2.waitKey(1)


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

    run_mv_tracking('Warehouse_017', opt)
    run_mv_tracking('Warehouse_018', opt)
    run_mv_tracking('Warehouse_019', opt)
    run_mv_tracking('Warehouse_020', opt)

    return

if __name__ == '__main__':
    main_multiview()
