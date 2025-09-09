import numpy as np
from scipy.spatial.distance import cdist
import cv2
from shapely.geometry import Point, Polygon
import os
import json

def generate_nested_folders(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)

def load_calib_infos(dataset_root, scene_name):
    # with open(dataset_root + '/' + calib_fol + '/' + scene_name + '.json', 'r') as json_file:
    with open(dataset_root + '/test/' + scene_name + '/calibration_modified.json', 'r') as json_file:
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

def estimate_overlap_ratio(box1, box2): # 1 to 2
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])


    return inter_area / area1

    # area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # union_area = area1 + area2 - inter_area

    # return inter_area / union_area if union_area > 0 else 0

def get_overlapping_indices(boxes, iou_thresh=0.85):
    """
    Find all pairs of box indices with IoU >= threshold.

    Args:
        boxes: np.ndarray of shape (N, 4) — format [x1, y1, x2, y2]
        iou_thresh: float — IoU threshold

    Returns:
        List of tuples: [(i, j), ...] where box i and box j overlap
    """
    N = len(boxes)
    overlaps = []

    for i in range(N):
        for j in range(N):
            if i != j:
                iou = estimate_overlap_ratio(boxes[i][1:5], boxes[j][1:5])
                if iou >= iou_thresh:
                    overlaps.append((i))
                    break

    return overlaps

def load_files_and_divide_per_frame_idx(fol_pth, split_idx_on_frameID):

    print('Loading ' + fol_pth)
    files = [f for f in os.listdir(fol_pth) if os.path.isfile(os.path.join(fol_pth, f))]

    frameID_to_files = defaultdict(list)
    for img_file in tqdm(files):
        fid = img_file.split('_')[split_idx_on_frameID]  # extract frameID from filename
        frameID_to_files[fid].append(img_file)

    return frameID_to_files


def get_pose_infor_from_path(a_person_pose_pth, person_det_info): # Load pose keypoint and shift the location to the image coordinate system
    with open(a_person_pose_pth, 'r') as json_file:
        pose_info = json.load(json_file)
        kpsc = np.array(pose_info[0]['keypoints'])
        kpsc[:, 0] += person_det_info[1]
        kpsc[:, 1] += person_det_info[2]
        return kpsc


def get_camera_keys(dataset_root, imgs_fol, scene_name):
    path = dataset_root + imgs_fol +  scene_name 
    folders = [f.split('_')[1] for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return sorted(folders)


def visualize_pose(img, pose, pcolor, kp_thresh=0.5):
    for kp in pose:
        if kp[2] > kp_thresh:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 5, pcolor, 2, cv2.LINE_AA)
    return img


def non_overlapped_region_dets(dets):
    non_overlaped_regions = np.full((1080, 1920), -1)

    for d_idx in range(len(dets)):
        bbox_tlwh = dets[d_idx][3:7]

        det_roi = non_overlaped_regions[bbox_tlwh[1] : bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0] : bbox_tlwh[0] + bbox_tlwh[2]]

        free_space = det_roi == -1
        non_free_space = det_roi >= 0


        det_roi[free_space] = d_idx
        det_roi[non_free_space] = -2

        non_overlaped_regions[bbox_tlwh[1] : bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0] : bbox_tlwh[0] + bbox_tlwh[2]] = det_roi

    # non_overlaped_regions_viz = np.zeros((1080, 1920, 3), dtype=np.int8)

    # for d_idx in range(len(dets)):
    #     color = tuple(np.random.randint(0, 256, size=3).tolist())
    #     non_overlaped_regions_viz[non_overlaped_regions==d_idx] = color

    # cv2.imshow('non-overlapped', non_overlaped_regions_viz)
    # cv2.waitKey(0)

    return non_overlaped_regions
        

def point_in_roi(px, py, x, y, w, h):
    return (x <= px < x + w) and (y <= py < y + h)


def cvt_camview_2_bev_with_depthmap(point2d, intrinsic, extrinsic, depth_value):

    K = intrinsic
    Z = depth_value / 1000

    # Extract R and t
    R = extrinsic[:3, :3]       # 3x3 rotation matrix
    t = extrinsic[:3, 3]        # 3x1 translation vector

    p = np.array([point2d[0], point2d[1], 1])
    p_norm = np.linalg.inv(K) @ p

    P_camera = Z * p_norm

    P_world = R.T @ (P_camera - t)

    return P_world[:2]


def cvt_camview_2_bev_with_depthmap_multi(points_2d, intrinsic, extrinsic, depth_value):

    K = intrinsic
    # Z = depth_value / 1000

    flatten_dm = depth_value.reshape(-1, 1)
    flatten_dm = flatten_dm / 1000.0

    # Extract R and t
    R = extrinsic[:3, :3]       # 3x3 rotation matrix
    t = extrinsic[:3, 3]        # 3x1 translation vector

    p2d_3x1 = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
    p_norm = np.linalg.inv(K) @ p2d_3x1.T

    P_camera = (p_norm.T * flatten_dm).T
    P_world = (R.T @ ((P_camera.T - t).T)).T

    return P_world


def get_optimize_dm_for_an_obj(bev_vkp_list):
    bev_vkp_dist = cdist(bev_vkp_list, bev_vkp_list)
    total_dists = bev_vkp_dist.sum(axis=1)
    best_idx = np.argmin(total_dists)
    return best_idx


def estimate_person_bev_location_with_depthmap(dets, poses, bev_dm, kp_thresh=0.5):

    apprx_bev = np.zeros((len(dets), 3))
    apprx_bev_flag = np.zeros(len(dets))

    for idx in range(poses.shape[0]):
        pose = poses[idx]
        # bbox_tlwh = dets[idx][3:7]

        if not pose is None:
            valid_pose_kps = pose[pose[:,2] > kp_thresh]
            if valid_pose_kps.shape[0] > 0:
                bev_vkp_list = []
                for vkp in valid_pose_kps:
                    if point_in_roi(vkp[0], vkp[1], 0, 0, 1920, 1080):
                        bev_vkp = bev_dm[int(vkp[1])][int(vkp[0])]
                        bev_vkp_list.append(bev_vkp)
                if len(bev_vkp_list) > 0:
                    bev_vkp_list = np.array(bev_vkp_list)
                    best_idx = get_optimize_dm_for_an_obj(bev_vkp_list)
                    bev_p = bev_vkp_list[best_idx]
                    apprx_bev[idx] = bev_p
                    apprx_bev_flag[idx] = 1

    dets_w_bev = np.concatenate((dets, apprx_bev), axis=1)
    dets_w_bev = np.concatenate((dets_w_bev, apprx_bev_flag.reshape(-1, 1)), axis=1)

    return dets_w_bev


# Can ultilize using median_z more efficient
def estimate_nov_tra_bev_location_with_depthmap(dets, bev_dm):

    apprx_bev = np.zeros((len(dets), 3))
    apprx_bev_flag = np.zeros(len(dets))

    for idx in range(dets.shape[0]):
        bbox_tlwh = dets[idx][3:7]
        # bbox = [bbox_tlwh[0], bbox_tlwh[1], bbox_tlwh[0] + bbox_tlwh[2], bbox_tlwh[1] + bbox_tlwh[3]]

        candidate_list = bev_dm[bbox_tlwh[1] : bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0] : bbox_tlwh[0] + bbox_tlwh[2], :]
        candidate_list = candidate_list.reshape(-1, 3)
        # best_idx = get_optimize_dm_for_an_obj(candidate_list)
        # center_x = bbox_tlwh[0] + bbox_tlwh[2] / 2
        # center_y = bbox_tlwh[1] + bbox_tlwh[3] / 2

        median_x = np.median(candidate_list[:, 0])
        median_y = np.median(candidate_list[:, 1])
        median_z = np.median(candidate_list[:, 2])
        
        apprx_bev[idx] = [median_x, median_y, median_z]
        apprx_bev_flag[idx] = 1

    dets_w_bev = np.concatenate((dets, apprx_bev), axis=1)
    dets_w_bev = np.concatenate((dets_w_bev, apprx_bev_flag.reshape(-1, 1)), axis=1)

    return dets_w_bev


# def estimate_fou_agi_bev_location_with_depthmap(dets, intrin, extrin, depthmap, z_threshold=1.1):
def estimate_fou_agi_bev_location_with_depthmap(dets, bev_dm, z_threshold=1.1):

    apprx_bev = np.zeros((len(dets), 3))
    apprx_bev_flag = np.zeros(len(dets))

    for idx in range(dets.shape[0]):
        bbox_tlwh = dets[idx][3:7]

        dm_roi = bev_dm[bbox_tlwh[1] : bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0] : bbox_tlwh[0] + bbox_tlwh[2], :]
        dm_roi = dm_roi.reshape(-1, 3)

        filtered_coords_3d = dm_roi[dm_roi[:, 2] > z_threshold]

        if filtered_coords_3d.shape[0] > 0:
            apprx_bev_flag[idx] = 1
        else:
            filtered_coords_3d = dm_roi[dm_roi[:, 2] > 0.05]
            if filtered_coords_3d.shape[0] > 0:
                apprx_bev_flag[idx] = 2

        if apprx_bev_flag[idx] > 0:
            best_idx = get_optimize_dm_for_an_obj(filtered_coords_3d)
            apprx_bev[idx] = filtered_coords_3d[best_idx]
        

    dets_w_bev = np.concatenate((dets, apprx_bev), axis=1)
    dets_w_bev = np.concatenate((dets_w_bev, apprx_bev_flag.reshape(-1, 1)), axis=1)

    return dets_w_bev


# def estimate_forklift_bev_location_with_depthmap(dets, intrin, extrin, depthmap, z_min=0.1, z_max=0.5):
def estimate_forklift_bev_location_with_depthmap(dets, bev_dm, z_min=0.1, z_max=0.5):

    apprx_bev = np.zeros((len(dets), 3))
    apprx_bev_flag = np.zeros(len(dets))

    for idx in range(dets.shape[0]):
        bbox_tlwh = dets[idx][3:7]

        # Generate all (x, y) coordinate pairs in the RoI
        yy, xx = np.meshgrid(np.arange(bbox_tlwh[1], bbox_tlwh[1] + bbox_tlwh[3]), 
                                np.arange(bbox_tlwh[0], bbox_tlwh[0] + bbox_tlwh[2]), indexing='ij')
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # shape: (N, 2)

        dm_roi = bev_dm[bbox_tlwh[1]:bbox_tlwh[1] + bbox_tlwh[3], bbox_tlwh[0]:bbox_tlwh[0]+bbox_tlwh[2], :]
        
        min_indices = dm_roi[:, 2] > z_min
        max_indices = dm_roi[:, 2] < z_max
        filtered_indices = np.where( min_indices & max_indices)[0]
        filtered_coords_3d = dm_roi[filtered_indices]

        if filtered_coords_3d.shape[0] > 0:
            apprx_bev_flag[idx] = 1
            filtered_coords_3d = filtered_coords_3d.reshape(-1, 3)
            best_idx = get_optimize_dm_for_an_obj(filtered_coords_3d)
            apprx_bev[idx] = filtered_coords_3d[best_idx]
        

    dets_w_bev = np.concatenate((dets, apprx_bev), axis=1)
    dets_w_bev = np.concatenate((dets_w_bev, apprx_bev_flag.reshape(-1, 1)), axis=1)

    return dets_w_bev


def cvt_dm_value_to_bev(intrin, extrin, depth_map):
    h, w = depth_map.shape
    yy, xx = np.meshgrid(np.arange(0, h), 
                        np.arange(0 + w), indexing='ij')
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # shape: (N, 2)

    bev_dm = cvt_camview_2_bev_with_depthmap_multi(coords, intrin, extrin, depth_map)

    return bev_dm


def convert_dets_cam_to_bev_w_depthmap(dets, poses, calib_info, depthmap):
    # h_mat = np.array(calib_info['homography'])
    intrinsic = np.array(calib_info['intrinsicMatrix'])
    extrinsic = np.array(calib_info['extrinsicMatrix'])


    bev_dm = cvt_dm_value_to_bev(intrinsic, extrinsic, depthmap)
    bev_dm = bev_dm.reshape((1080, 1920, 3))

    # For person
    person_indices = dets[:, 1] == 0
    person_dets = dets[person_indices]
    person_poses = poses[person_indices]
    # person_dets_w_bev = estimate_person_bev_location_with_depthmap(person_dets, person_poses, 
    #     intrinsic, extrinsic, depthmap, kp_thresh=0.5)
    person_dets_w_bev = estimate_person_bev_location_with_depthmap(person_dets, person_poses, 
        bev_dm, kp_thresh=0.5)


    nov_tra_indices = np.where( (dets[:, 1] == 2) | (dets[:, 1] == 3) )[0]
    nov_tra_dets = dets[nov_tra_indices]
    # nov_tra_dets_w_bev = estimate_nov_tra_bev_location_with_depthmap(nov_tra_dets, intrinsic, extrinsic, depthmap)
    nov_tra_dets_w_bev = estimate_nov_tra_bev_location_with_depthmap(nov_tra_dets, bev_dm)


    # FourrierGR1T2 + AgilityDigit
    fou_agi_indices = np.where( (dets[:, 1] == 4) | (dets[:, 1] == 5) )[0]
    fou_agi_dets = dets[fou_agi_indices]
    # fou_agi_dets_w_bev = estimate_fou_agi_bev_location_with_depthmap(fou_agi_dets, intrinsic, extrinsic, depthmap, z_threshold=1.1)
    fou_agi_dets_w_bev = estimate_fou_agi_bev_location_with_depthmap(fou_agi_dets, bev_dm, z_threshold=1.1)


    # Forklift
    forklift_indices = dets[:, 1] == 1
    forklift_dets = dets[forklift_indices]
    # forklift_dets_w_bev = estimate_forklift_bev_location_with_depthmap(forklift_dets, intrinsic, extrinsic, 
    #                                     depthmap, z_min=0.1, z_max=0.5)
    forklift_dets_w_bev = estimate_forklift_bev_location_with_depthmap(forklift_dets, bev_dm, z_min=0.1, z_max=0.5)

    dets_w_bev = np.concatenate((person_dets_w_bev, nov_tra_dets_w_bev), axis=0)
    dets_w_bev = np.concatenate((dets_w_bev, fou_agi_dets_w_bev), axis=0)
    dets_w_bev = np.concatenate((dets_w_bev, forklift_dets_w_bev), axis=0)

    return dets_w_bev


def estimate_diff_cam_matrix_dist(dets):

    det_len = len(dets)
    diff_cam_mat = np.zeros((det_len, det_len))

    for idx_i in range(det_len):
        for idx_j in range(idx_i, det_len):
            if dets[idx_i][0] == dets[idx_j][0]:
                diff_cam_mat[idx_i][idx_j] = 1.0
                diff_cam_mat[idx_j][idx_i] = 1.0

    return diff_cam_mat


def estimate_bev_matrix_dist(dets):
    bev_dets = dets[:, 8:10]
    bev_dist_mat = cdist(bev_dets, bev_dets)
    return bev_dist_mat


def estimate_bev_matrix_dist_diff_det_set(dets1, dets2):
    bev_dist_mat = cdist(dets1, dets2)
    return bev_dist_mat


def normalize_distance(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)  # avoid division by zero

    return arr_norm


def cosine_dist(a, b):
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - np.matmul(a_norm, b_norm.T)  # shape: (N, M)


def estimate_embedding_distance_kpr(feats, visses, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(feats), len(feats)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    # det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    # # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    # track_features = np.asarray([track.curr_feat for track in tracks], dtype=np.float64)

    # det_viss = np.asarray([track.curr_viss for track in detections], dtype=np.float64)
    # # track_viss = np.asarray([track.smooth_viss for track in tracks], dtype=np.float64)
    # track_viss = np.asarray([track.curr_viss for track in tracks], dtype=np.float64)

    body_part_dists = [] # for 6 body parts
    for p in range(6):
        part_dist = cosine_dist(feats[:, p, :], feats[:, p, :])  # shape: (N, M)
        body_part_dists.append(part_dist)

    body_part_dists = np.stack(body_part_dists, axis=0)  # shape: (6, N, M)

    # Validity mask: shape (6, N, M)
    valid_mask = np.sqrt(visses.T[:, :, None] * visses.T[:, None, :])

    masked_dists = body_part_dists * valid_mask
    mean_dists = np.sum(masked_dists, axis=0) / (np.sum(valid_mask, axis=0) + 1e-8)

    return mean_dists


def estimate_embedding_distance_kpr_2(query_feats, gallery_feats, 
                                        query_visses, gallery_visses,
                                            metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(query_feats), len(gallery_feats)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    # det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    # # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    # track_features = np.asarray([track.curr_feat for track in tracks], dtype=np.float64)

    # det_viss = np.asarray([track.curr_viss for track in detections], dtype=np.float64)
    # # track_viss = np.asarray([track.smooth_viss for track in tracks], dtype=np.float64)
    # track_viss = np.asarray([track.curr_viss for track in tracks], dtype=np.float64)

    body_part_dists = [] # for 6 body parts
    for p in range(6):
        part_dist = cosine_dist(query_feats[:, p, :], gallery_feats[:, p, :])  # shape: (N, M)
        body_part_dists.append(part_dist)

    body_part_dists = np.stack(body_part_dists, axis=0)  # shape: (6, N, M)

    # Validity mask: shape (6, N, M)
    valid_mask = np.sqrt(query_visses.T[:, :, None] * gallery_visses.T[:, None, :])

    masked_dists = body_part_dists * valid_mask
    mean_dists = np.sum(masked_dists, axis=0) / (np.sum(valid_mask, axis=0) + 1e-8)

    return mean_dists


# cam_idx class_idx sv_track_idx x y w h ps_flag x y z bev_flag
def clustering_person(person_dets, person_poses, person_feats, person_visses, bev_dist_thresh=1.0, reid_dist_thresh=0.35):


    number_of_person = person_dets.shape[0]

    diff_cam_matrix_dist = estimate_diff_cam_matrix_dist(person_dets)
    bev_matrix_dist = estimate_bev_matrix_dist(person_dets)
    bev_matrix_dist = bev_matrix_dist < bev_dist_thresh
 

    reid_dist = estimate_embedding_distance_kpr(person_feats, person_visses)
    norm_reid_dist = reid_dist < reid_dist_thresh

    person_stt_mat = np.full(number_of_person, -1)
    person_groups = []
    for idx in range(number_of_person):
        bev_close_obj = np.where( (bev_matrix_dist[idx]) & (diff_cam_matrix_dist[idx]==0) )[0]
        if person_dets[idx][11] == 0:
            bev_close_obj = []
        
        reid_close_obj = np.where( (norm_reid_dist[idx]) & (diff_cam_matrix_dist[idx]==0) )[0]
        
        unified_close_obj = np.intersect1d(bev_close_obj, reid_close_obj)

        # person_groups.append(unified_close_obj.astype(np.int32))
        person_groups.append([idx])
        for linked_person in unified_close_obj:
            person_groups[-1].append(linked_person)
        
        # No match and have near neighbors
        if (len(unified_close_obj) == 0) and (len(bev_close_obj) > 0):
            person_stt_mat[idx] = -2

    global_person_idx = 0
    
    # Simple first match
    for p_idx in range(number_of_person):
        group_stt = person_stt_mat[person_groups[p_idx]]
        if (group_stt == -1).all():
            person_stt_mat[person_groups[p_idx]] = global_person_idx
            global_person_idx += 1

    # Second match
    for p_idx in range(number_of_person):
        if person_stt_mat[p_idx] == -1:
            corr_g_idx = np.unique(person_stt_mat[person_groups[p_idx][1:]])
            if len(corr_g_idx) == 1:
                person_stt_mat[p_idx] = corr_g_idx[0]

    # print(person_stt_mat)

    return person_stt_mat


def estimate_diff_class_matrix_dist(dets):

    det_len = len(dets)
    diff_class_mat = np.zeros((det_len, det_len))

    for idx_i in range(det_len):
        for idx_j in range(idx_i, det_len):
            if dets[idx_i][1] != dets[idx_j][1]:
                diff_class_mat[idx_i][idx_j] = 1.0
                diff_class_mat[idx_j][idx_i] = 1.0

    return diff_class_mat


def clustering_dets_bev_only(other_dets, mv_dets=None, bev_dist_thresh=1.0):
    number_of_dets = other_dets.shape[0]

    diff_cam_matrix_dist = estimate_diff_class_matrix_dist(other_dets)
    bev_matrix_dist = estimate_bev_matrix_dist(other_dets)
    bev_matrix_dist = bev_matrix_dist < bev_dist_thresh

    obj_groups = []
    for idx in range(number_of_dets):
        bev_close_obj = np.where( (bev_matrix_dist[idx]) & (diff_cam_matrix_dist[idx]==0) )[0]
        obj_groups.append([idx])
        for linked_obj in bev_close_obj:
            obj_groups[-1].append(linked_obj)
    
    obj_stt_mat = np.full(number_of_dets, -1)
    global_obj_idx = 0
    
    # Simple first match
    for p_idx in range(number_of_dets):
        group_stt = obj_stt_mat[obj_groups[p_idx]]
        if (group_stt == -1).all():
            obj_stt_mat[obj_groups[p_idx]] = global_obj_idx
            global_obj_idx += 1
    
    # Second match
    for p_idx in range(number_of_dets):
        if obj_stt_mat[p_idx] == -1:
            corr_g_idx = np.unique(obj_stt_mat[obj_groups[p_idx][1:]])
            if len(corr_g_idx) == 1:
                obj_stt_mat[p_idx] = corr_g_idx[0]

    mv_bev_det_list = []
    if not mv_dets is None:
        for mv_det in mv_dets:
            mv_bev_det_list.append(mv_det.bev_coor)
        mv_bev_det_arr = np.array(mv_bev_det_list)

        bev_dist_mat = estimate_bev_matrix_dist_diff_det_set(other_dets[:, 8:11], mv_bev_det_arr)
        for idx_od in range(len(other_dets)):
            if (np.min(bev_dist_mat[idx_od]) < bev_dist_thresh * 2):
                obj_stt_mat[idx_od] = -1
    
    return obj_stt_mat


def map_dets_to_tracked_objs_bev_only(other_dets, mv_dets=None, bev_dist_thresh=1.0):

    number_of_dets = other_dets.shape[0]
    obj_stt_mat = np.full(number_of_dets, -1)

    mv_bev_det_list = []
    if not mv_dets is None:
        for mv_det in mv_dets:
            mv_bev_det_list.append(mv_det.bev_coor)
        mv_bev_det_arr = np.array(mv_bev_det_list)

        bev_dist_mat = estimate_bev_matrix_dist_diff_det_set(other_dets[:, 8:11], mv_bev_det_arr)
        # print(bev_dist_mat)
        for idx_od in range(bev_dist_mat.shape[0]):
            det_camIdx = other_dets[idx_od][0]
            for idx_mv in range(bev_dist_mat.shape[1]):
                if bev_dist_mat[idx_od][idx_mv] < bev_dist_thresh and mv_dets[idx_mv].linked_cam[det_camIdx] == -1:
                    obj_stt_mat[idx_od] = idx_mv
        
        unique_match_mv_indices = np.unique(obj_stt_mat)
        for uniq_match_mv_index in unique_match_mv_indices:
            if uniq_match_mv_index >= 0:
                potential_dets = other_dets[obj_stt_mat == uniq_match_mv_index]
                potential_dets_BEV_dist_to_candidate = bev_dist_mat[obj_stt_mat == uniq_match_mv_index][:, uniq_match_mv_index]
                # if len(np.unique(potential_dets[:, 0])) == len(potential_dets): # Different camidx
    
    return obj_stt_mat


def check_dets_not_from_same_cam_and_close_to_each_other(dets, bev_dist_thresh):
    # cam_idx class_idx sv_track_idx x y w h ps_flag x y z bev_flag

    cam_idx_list = dets[:, 0]
    if len(np.unique(cam_idx_list)) != len(dets):
        # print('CamIdx probelm')
        return False

    bev_list = dets[:, 8:10]
    bev_dist_mat = estimate_bev_matrix_dist(bev_list)
    if np.max(bev_dist_mat) > bev_dist_thresh:
        # print('BeV Prob')
        return False
    
    return True


def bev_dist_between_2_mv_objs(mv1, mv2):
    x_dist = mv1.bev_coor[0] - mv2.bev_coor[0]
    y_dist = mv1.bev_coor[1] - mv2.bev_coor[1]
    
    return np.sqrt(x_dist**2 + y_dist**2)
    

def visualize_clusters(dets, group_idx, cam_imgs):
    unique_group_id = np.unique(group_idx)

    for p_id in unique_group_id:
        p_id_dets_indices = group_idx == p_id
        p_id_dets = dets[p_id_dets_indices]
        for adet in p_id_dets:
            adet_bbox = adet[3:7]
            adet_roi = cam_imgs[int(adet[0])][int(adet_bbox[1]) : int(adet_bbox[1]) + int(adet_bbox[3]), 
                                                int(adet_bbox[0]) : int(adet_bbox[0]) + int(adet_bbox[2]), : ]
            cv2.imshow(str(adet[0]) + '_' + str(adet[2]), adet_roi)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def remove_dets_based_on_dm_ratio(dets, dm_ratio_thresh=0.3):
    keep_det_indices = np.zeros(len(dets), dtype=np.bool_)

    for d_idx in range(len(dets)):

        d_area = dets[d_idx][5] * dets[d_idx][6]
        d_dm_ratio = dets[d_idx][13] / d_area

        if d_dm_ratio > dm_ratio_thresh:
            keep_det_indices[d_idx] = 1
    
    return keep_det_indices


def remove_dets_based_on_camID_and_regions(dets, rectangle_point_list, cam_id):
    keep_det_indices = np.ones(len(dets), dtype=np.bool_)

    unwanted_reg = Polygon(rectangle_point_list) # shoudl be a list of tuple

    for d_idx in range(len(dets)):
        det  = dets[d_idx]
        bev_p = (det[8], det[9])
        inside = unwanted_reg.contains(Point(bev_p)) or unwanted_reg.touches(Point(bev_p))

        if det[0] == cam_id and inside:
            keep_det_indices[d_idx] = 0
    
    return keep_det_indices


def compute_bev_dist(xyz1, xyz2):
    diff = np.abs(xyz1 - xyz2)
    dist = np.sqrt(diff[0]**2 + diff[1]**2)

    return dist
    

def find_all_BEV_close_for_classID(arr, class_id, bev_dist_thresh=0.2):
    xyzz = arr[:, 8:12]
    ids = arr[:, 1]

    # Convert to (x1, y1, x2, y2)

    N = arr.shape[0]
    overlaps = {}

    for i in range(N):
        if ids[i] != class_id or xyzz[i][3] == 0:
            continue

        i_xyz = xyzz[i, :2]
        current_matches = []

        for j in range(N):
            
            if i == j or xyzz[i][3] == 0 or (ids[i] == ids[j]):
                continue

            j_xyz = xyzz[j, :2]
            
            ij_dist = compute_bev_dist(i_xyz, j_xyz)

            if ij_dist < bev_dist_thresh:
                current_matches.append(j)

        if current_matches:
            overlaps[i] = current_matches

    return overlaps


def remove_overlapped_four_agi(dets, camkeys, overlap_threshold=0.1):
    remove_indices = np.zeros(len(dets))

    for camidx in range(len(camkeys)):
        same_cam_idx = dets[:, 0]==camidx
        person_obj_idx = dets[:, 1]== 0
        four_obj_idx = dets[:, 1]== 4
        agi_obj_idx = dets[:, 1]== 5

        person_four_agi_obj_idx = person_obj_idx | four_obj_idx | agi_obj_idx

        cam_dets_indices = np.where(same_cam_idx & person_four_agi_obj_idx)[0]

        overlaps = find_all_BEV_close_for_classID(dets[cam_dets_indices], 5, overlap_threshold)
        # overlaps_4 = find_all_overlaps_for_classID(dets[cam_dets_indices], 4, overlap_threshold)
        # overlaps_5 = find_all_overlaps_for_classID(dets[cam_dets_indices], 5, overlap_threshold)

        if (len(overlaps)):
            print(camidx)
            # print(dets[cam_dets_indices])
            print('overlaps: ')
            print(overlaps)
            for overlap in overlaps:
                overlap_idx = cam_dets_indices[overlap]
                print(dets[overlap_idx][1], end=' ')
                for item in overlaps[overlap]:
                    item_idx = cam_dets_indices[item]
                    print(dets[item_idx][1])

            print(' ')

            break


def check_bev_coor_of_dets(dets, cam_calibs, camera_keys):

    def project_xyz_to_image(points, K, Rt):
        """
        Projects 3D points to 2D image coordinates using camera intrinsics and extrinsics.

        Args:
            points: (N, 3) array of 3D points in world/LiDAR coordinates
            K: (3, 3) camera intrinsic matrix
            Rt: (4, 4) camera extrinsic matrix (world → camera transform)

        Returns:
            pixels: (N, 2) array of 2D image coordinates
        """
        N = points.shape[0]

        # Convert to homogeneous coordinates: (N, 4)
        points_h = np.hstack((points, np.ones((N, 1))))  # (N, 4)

        # Transform to camera coordinates: (N, 4) x (4, 4).T → (N, 4)
        cam_points = (Rt @ points_h.T).T[:, :3]  # (N, 3)

        # Apply intrinsic matrix: (N, 3) x (3, 3).T → (N, 3)
        img_coords_h = (K @ cam_points.T).T  # (N, 3)

        # Convert to 2D image coordinates (u, v) by dividing by z
        pixels = img_coords_h[:, :2] / (img_coords_h[:, 2:3] + 1e-8)

        return pixels

    def points_in_bboxes(points, bboxes):
        """
        Args:
            points: (N, 2) array of points (u, v)
            bboxes: (N, 4) array of bounding boxes (x, y, w, h)

        Returns:
            inside_mask: (N,) boolean array — True if point i is inside bbox i
        """
        x, y = points[:, 0], points[:, 1]
        bx, by, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        inside_x = (x >= bx) & (x <= bx + bw)
        inside_y = (y >= by) & (y <= by + bh)

        return inside_x & inside_y

    not_good_det_indices = []

    for camIdx in range(len(camera_keys)):
        cam_calib = cam_calibs[camera_keys[camIdx]]
        intrinsic = np.array(cam_calib['intrinsicMatrix'])
        extrinsic = np.array(cam_calib['extrinsicMatrix'])

        cam_det_indices = np.where(dets[:, 0]==camIdx)[0]
        cam_dets = dets[cam_det_indices]

        bev_cam_dets = cam_dets[:, 8:11]
        bbox_cam_dets = cam_dets[:, 3:7]
        cam_points = project_xyz_to_image(bev_cam_dets, intrinsic, extrinsic)
        inside_mat = points_in_bboxes(cam_points, bbox_cam_dets)

        if len(not_good_det_indices):
            np.concatenate((not_good_det_indices, cam_det_indices[np.where(inside_mat == False)[0]]))
        else:
            not_good_det_indices = cam_det_indices[np.where(inside_mat == False)[0]]

    not_good_det_indices = not_good_det_indices.reshape(-1)
    if len(not_good_det_indices):
        print(not_good_det_indices)


def remove_too_tight_dets(dets, classID_list, w_threshold=20):
    keep_det_indices = np.ones(len(dets), dtype=np.bool_)
    for classID in classID_list:
        class_det_indices = np.where(dets[:, 1]==classID)[0]
        class_dets = dets[class_det_indices]

        small_w_det_indices = np.where(class_dets[:, 5] < w_threshold)[0]
        keep_det_indices[class_det_indices[small_w_det_indices]] = 0
    return keep_det_indices
    