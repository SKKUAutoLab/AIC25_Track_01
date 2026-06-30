import cv2
import os
import numpy as np
import argparse
import json
from tqdm import tqdm

import pita_utils as my_utils
import h5py

import configuration as config
import open3d as o3d


max_frame_idx = config.TOTAL_FRAME
dataset_root = config.ROOT_DATA_FOLDER
imgs_fol = config.FULL_IMAGE_FOLDER

sub_folder_tracking_results = config.RESULTS_FOLDER
track_sv_info_fol = config.SV_TRACK_RESULTS
static_dm_fol = config.STATIC_DM_FOL
dynamic_dm_fol = config.DYNAMIC_DM_FOL

SCENES = ['Warehouse_017', 'Warehouse_018', 'Warehouse_019', 'Warehouse_020']


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_results_from_singview_1cam(result_pth, camkeys, camkey, frameIdx):
    dets_arr = []
    pose_arr = []
    feat_arr = []
    viss_arr = []

    fake_pose = np.zeros((17, 3))
    fake_feat = np.zeros((6, 512))
    fake_viss = np.zeros((6))

    for idx in range(len(camkeys)):
        cur_camkey = camkeys[idx]
        if cur_camkey == camkey:
            with open(result_pth[idx] + '/' + str(frameIdx).zfill(7) + '.json', 'r') as json_file:
                sv_result_idx = json.load(json_file)
                for obj in sv_result_idx:
                    tid = obj['id']
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


def _load_scene_context(scene_name):
    camera_keys = my_utils.get_camera_keys(dataset_root=dataset_root, imgs_fol=imgs_fol, scene_name=scene_name)
    print('Camkey: ', camera_keys)
    calib_info = my_utils.load_calib_infos(dataset_root=dataset_root, scene_name=scene_name)
    return camera_keys, calib_info


def generate_dm_exclude(scene_name, args):
    print('Run generate_dm_exclude on scene: ' + scene_name)
    camera_keys, calib_info = _load_scene_context(scene_name)
    exclude_dm_list = _prepare_static_dm_folders(scene_name, camera_keys)
    frame_step = config.FRAME_STEP
    for camIdx in tqdm(range(len(camera_keys)), desc='Camera'):
        camkey = camera_keys[camIdx]
        exclude_depth_map = _compute_exclude_depth_map(scene_name, camkey, frame_step, max_frame_idx)
        _save_exclude_depth_map(exclude_dm_list[camIdx], exclude_depth_map)


def _prepare_static_dm_folders(scene_name, camera_keys):
    exclude_dm_list = []
    for i in range(len(camera_keys)):
        exclude_dm_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + static_dm_fol + '/Camera_' + camera_keys[i]
        exclude_dm_list.append(exclude_dm_pth)
        my_utils.generate_nested_folders(exclude_dm_pth)
    return exclude_dm_list


def _compute_exclude_depth_map(scene_name, camkey, frame_step, max_frame):
    cam_depth_map_pth = dataset_root + 'test/' + scene_name + '/depth_maps/Camera_' + camkey + '.h5'
    total_depth_map = np.zeros((int(max_frame / frame_step), 1080, 1920))
    with h5py.File(cam_depth_map_pth, 'r') as f:
        for frame_idx in tqdm(range(0, max_frame, frame_step), desc=scene_name + '_' + camkey):
            depth_map_key = 'distance_to_image_plane_' + str(frame_idx).zfill(5) + '.png'
            total_depth_map[int(frame_idx / frame_step)] = np.array(f[depth_map_key])
    print('Evaluating ' + scene_name + '_' + camkey)
    exclude_depth_map = np.zeros((1080, 1920))
    for i in range(1080):
        for j in range(1920):
            arr = total_depth_map[:, i, j].reshape(-1)
            unique_vals, counts = np.unique(arr, return_counts=True)
            sorted_idx = np.argsort(-counts)
            exclude_depth_map[i][j] = unique_vals[sorted_idx][0]
    return exclude_depth_map


def _save_exclude_depth_map(exclude_dm_folder, exclude_depth_map):
    np.save(exclude_dm_folder + '/exclude_dm.npy', exclude_depth_map)
    depth_map_norm = cv2.normalize(exclude_depth_map.copy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map_unit8 = depth_map_norm.astype(np.uint8)
    cv2.imwrite(exclude_dm_folder + '/exclude_dm.jpg', depth_map_unit8)


def generate_bev_dm(scene_name, args):
    print('Run generate_bev_dm on scene: ' + scene_name)
    camera_keys, calib_info = _load_scene_context(scene_name)
    sv_result_list, exclude_dm_list, bev_dm_list = _prepare_bev_folders(scene_name, camera_keys)
    max_frame = config.MAX_FRAME
    for camIdx in tqdm(range(len(camera_keys)), desc='Camera'):
        camkey = camera_keys[camIdx]
        cam_calib = calib_info[camkey]
        cam_depth_map_pth = dataset_root + 'test/' + scene_name + '/depth_maps/Camera_' + camkey + '.h5'
        cam_exclude_dm_arr = np.load(exclude_dm_list[camIdx] + '/exclude_dm.npy')
        with h5py.File(cam_depth_map_pth, 'r') as f:
            for frame_idx in tqdm(range(max_frame), desc=scene_name + '_' + camkey):
                save_dict = _build_bev_frame(f, frame_idx, cam_calib, cam_exclude_dm_arr, sv_result_list, camera_keys, camkey)
                json.dump(save_dict,
                        open(bev_dm_list[camIdx] + '/' + str(frame_idx).zfill(7) + '.json', 'w'),
                        indent=4, cls=NumpyEncoder)


def _prepare_bev_folders(scene_name, camera_keys):
    sv_result_list = []
    exclude_dm_list = []
    bev_dm_list = []
    for i in range(len(camera_keys)):
        sv_result_list.append(dataset_root + sub_folder_tracking_results + scene_name + '/' + track_sv_info_fol + '/Camera_' + camera_keys[i])
        exclude_dm_list.append(dataset_root + sub_folder_tracking_results + scene_name + '/' + static_dm_fol + '/Camera_' + camera_keys[i])
        bev_dm_pth = dataset_root + sub_folder_tracking_results + scene_name + '/' + dynamic_dm_fol + '/Camera_' + camera_keys[i]
        bev_dm_list.append(bev_dm_pth)
        my_utils.generate_nested_folders(bev_dm_pth)
    return sv_result_list, exclude_dm_list, bev_dm_list


def _build_bev_frame(h5_file, frame_idx, cam_calib, cam_exclude_dm_arr, sv_result_list, camera_keys, camkey):
    depth_map_key = 'distance_to_image_plane_' + str(frame_idx).zfill(5) + '.png'
    depth_map = np.array(h5_file[depth_map_key])
    diff_dm = depth_map - cam_exclude_dm_arr
    diff_dm = (diff_dm != 0).astype(int)
    depth_map = depth_map * diff_dm
    intrinsic = np.array(cam_calib['intrinsicMatrix'])
    extrinsic = np.array(cam_calib['extrinsicMatrix'])
    bev_dm = my_utils.cvt_dm_value_to_bev(intrinsic, extrinsic, depth_map)
    bev_dm = bev_dm.reshape((1080, 1920, 3)).astype(np.float32)
    dets, poses, feats, visses = load_results_from_singview_1cam(sv_result_list, camera_keys, camkey, frame_idx)
    if len(dets) > 0:
        dets = _append_bev_coordinates(dets, bev_dm, diff_dm)
    return _bev_save_dict(dets, poses, feats, visses)


def _append_bev_coordinates(dets, bev_dm, diff_dm):
    non_overlapped_regions = my_utils.non_overlapped_region_dets(dets)
    bev_arr_list = np.zeros((len(dets), 6))
    for d_idx in range(len(dets)):
        bbox_tlwh = dets[d_idx][3:7]
        bev_dm_roi = bev_dm[bbox_tlwh[1]:bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0]:bbox_tlwh[0] + bbox_tlwh[2], :]
        exclude_dm_roi = diff_dm[bbox_tlwh[1]:bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0]:bbox_tlwh[0] + bbox_tlwh[2]]
        non_overlapped_det = non_overlapped_regions[bbox_tlwh[1]:bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0]:bbox_tlwh[0] + bbox_tlwh[2]]
        obj_bev_dm = bev_dm_roi[(exclude_dm_roi != 0) & (non_overlapped_det == d_idx)]
        if obj_bev_dm.shape[0] > 0:
            bev_arr_list[d_idx][0] = np.mean(obj_bev_dm[:, 0])
            bev_arr_list[d_idx][1] = np.mean(obj_bev_dm[:, 1])
            bev_arr_list[d_idx][2] = np.mean(obj_bev_dm[:, 2])
            bev_arr_list[d_idx][3] = 1.0
            bev_arr_list[d_idx][4] = np.max(obj_bev_dm[:, 2])
            bev_arr_list[d_idx][5] = obj_bev_dm.shape[0]
    bev_arr = np.array(bev_arr_list)
    return np.concatenate((dets, bev_arr), axis=1)


def _bev_save_dict(dets, poses, feats, visses):
    if len(dets):
        return {'det_info': dets, 'poses': poses, 'feats': feats, 'visses': visses}
    return {'det_info': None, 'poses': None, 'feats': None, 'visses': None}


def generate_pointcloud_from_dynamic_dm(scene_name):
    print('Generate point cloud from dynamic depth maps for scene: ' + scene_name)
    camera_keys, calib_info = _load_scene_context(scene_name)
    pointcloud_output_dir = os.path.join(dataset_root, sub_folder_tracking_results, scene_name, 'open3d')
    my_utils.generate_nested_folders(pointcloud_output_dir)
    max_frame = config.MAX_FRAME
    for frame_idx in tqdm(range(max_frame), desc=f'Processing frames for {scene_name}'):
        all_points = []
        for camIdx in range(len(camera_keys)):
            camkey = camera_keys[camIdx]
            points_world = _camera_world_points(scene_name, camkey, calib_info[camkey], frame_idx)
            if points_world is not None:
                all_points.append(points_world)
        if all_points:
            _save_pointcloud(pointcloud_output_dir, frame_idx, all_points)


def _camera_world_points(scene_name, camkey, cam_calib, frame_idx):
    cam_depth_map_pth = os.path.join(dataset_root, 'test', scene_name, 'depth_maps', f'Camera_{camkey}.h5')
    cam_exclude_dm_pth = os.path.join(dataset_root, sub_folder_tracking_results, scene_name, static_dm_fol, f'Camera_{camkey}', 'exclude_dm.npy')
    cam_exclude_dm_arr = np.load(cam_exclude_dm_pth)
    with h5py.File(cam_depth_map_pth, 'r') as f:
        depth_map_key = f'distance_to_image_plane_{str(frame_idx).zfill(5)}.png'
        depth_map = np.array(f[depth_map_key])
        diff_dm = depth_map - cam_exclude_dm_arr
        dynamic_mask = (diff_dm != 0)
        dynamic_depth_map = depth_map * dynamic_mask.astype(int)
        dynamic_depth_map = dynamic_depth_map / 1000.0
        intrinsic = np.array(cam_calib['intrinsicMatrix'])
        extrinsic = np.array(cam_calib['extrinsicMatrix'])
        height, width = dynamic_depth_map.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        valid_mask = (dynamic_depth_map > 0) & dynamic_mask
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_depth = dynamic_depth_map[valid_mask]
        if len(valid_depth) == 0:
            return None
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x_cam = (valid_u - cx) * valid_depth / fx
        y_cam = (valid_v - cy) * valid_depth / fy
        z_cam = valid_depth
        points_cam_h = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)
        extrinsic_np = np.array(extrinsic, dtype=np.float64)
        if extrinsic_np.shape == (3, 4):
            extrinsic_np = np.vstack([extrinsic_np, np.array([0.0, 0.0, 0.0, 1.0])])
        cam_to_world = np.linalg.inv(extrinsic_np)
        return (cam_to_world @ points_cam_h.T).T[:, :3]


def _save_pointcloud(pointcloud_output_dir, frame_idx, all_points):
    combined_points = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    output_path = os.path.join(pointcloud_output_dir, f'{str(frame_idx).zfill(5)}.ply')
    o3d.io.write_point_cloud(output_path, pcd)


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


def main():
    opt = make_parser_multiview()
    for scene_name in SCENES:
        generate_dm_exclude(scene_name, opt)
    for scene_name in SCENES:
        generate_bev_dm(scene_name, opt)
    for scene_name in SCENES:
        generate_pointcloud_from_dynamic_dm(scene_name)


if __name__ == '__main__':
    main()
