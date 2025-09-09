import cv2
import numpy as np
from scipy.spatial.distance import cdist

from collections import deque

import pita_utils

class ID_Generator():
    def __init__(self):
        self.cur_id = 0
    
    def get_cur_id(self):
        cur_id = self.cur_id
        self.cur_id += 1
        return cur_id


class Multiview_Person():

    def __init__(self, camkeys, dets, poses=None, feats=None, visses=None, frame_idx = -1, history_len=360):
        
        self.camkeys = camkeys

        self.obj_cls = dets[0][1]   # Always the same class on the multiple det input

        self.linked_cam = np.full(len(camkeys), -1)
        self.linked_bbox = np.zeros((len(camkeys), 4), dtype=np.float32)
        self.linked_bev = np.zeros((len(camkeys), 4), dtype=np.float32)
        self.linked_extra_infos = np.zeros((len(camkeys), 2), dtype=np.float32)
        self.linked_poses = np.zeros((len(camkeys), 17, 3), dtype=np.float32)
        self.linked_feats = np.zeros((len(camkeys), 6, 512), dtype=np.float32)
        self.linked_visses = np.zeros((len(camkeys), 6), dtype=np.float32)

        self.height_from_dm = -1.0

        self.bev_coor = None

        self.obj_stt = -1
        self.global_id = -1

        self.last_bev_coor = None
        self.last_frame_idx = -1

        self.dm_ratio_thresh = 0.3
        self.upper_body_pose_thresh = 0.5

        self.temp_indices = []
        self.temp_dets = []
        self.temp_feats = []
        self.temp_visses = []
        self.temp_poses = []

        self.hist_cam = deque([], maxlen=history_len)
        self.hist_bbox = deque([], maxlen=history_len)
        self.hist_bev = deque([], maxlen=history_len)
        self.hist_feats = deque([], maxlen=history_len)
        self.hist_visses = deque([], maxlen=history_len)
        self.hist_poses = deque([], maxlen=history_len)
        self.hist_heights = []

        self.history_len = history_len
        self.temp_bev_coor_during_missing = deque([], maxlen=history_len)

        self.init_update(dets, poses, feats, visses)
        self.estimate_bev_coor()

    def init_update(self, dets, poses, feats, visses):
        
        for d_idx in range(dets.shape[0]):
            cam_d_idx = int(dets[d_idx][0])
            self.linked_cam[cam_d_idx] = dets[d_idx][2]
            self.linked_bev[cam_d_idx] = dets[d_idx][8:12]
            self.linked_bbox[cam_d_idx] = dets[d_idx][3:7]
            self.linked_extra_infos[cam_d_idx] = dets[d_idx][12:14]

            if self.obj_cls == 0: # Person
                self.linked_poses[cam_d_idx] = poses[d_idx]
                self.linked_feats[cam_d_idx] = feats[d_idx]
                self.linked_visses[cam_d_idx] = visses[d_idx]

        available_camera_idx = np.where(self.linked_cam != -1)[0]
        for aci in available_camera_idx:
            upper_body_pose_confs = np.mean(self.linked_poses[aci][:, 2][:7])
            if (upper_body_pose_confs > self.upper_body_pose_thresh):
                self.hist_heights.append(self.linked_extra_infos[aci][0])

    def estimate_bev_coor(self):
        available_bev_coors = self.linked_bev[self.linked_bev[:, 3] == 1][:, :3]
        if available_bev_coors.shape[0] > 0:
            self.bev_coor = available_bev_coors.mean(axis=0)
        else:
            self.bev_coor = None

    def return_available_feats_visses(self):
        stacked_feats = []
        stacked_visses = []

        for camIdx in range(len(self.camkeys)):
            if self.linked_cam[camIdx] >= 0:
                stacked_feats.append(self.linked_feats[camIdx])
                stacked_visses.append(self.linked_visses[camIdx])
        
        return np.array(stacked_feats), np.array(stacked_visses)

    def update_using_single_det(self, det, pose, feat, viss, frameIdx):
        d1_det_cam_idx = int(det[0])
        self.linked_cam[d1_det_cam_idx] = det[2]
        self.linked_bbox[d1_det_cam_idx] = det[3:7]
        self.linked_bev[d1_det_cam_idx] = det[8:12]
        self.linked_extra_infos[d1_det_cam_idx] = det[12:14]

        if self.obj_cls == 0: # Person
            self.linked_poses[d1_det_cam_idx] = pose
            self.linked_feats[d1_det_cam_idx] = feat
            self.linked_visses[d1_det_cam_idx] = viss

        self.estimate_bev_coor()
        self.last_bev_coor = self.bev_coor
        self.last_frame_idx = frameIdx

    def update_using_mv_det(self, mv_det, frameIdx):
        self.linked_cam = mv_det.linked_cam
        self.linked_bbox = mv_det.linked_bbox
        self.linked_bev = mv_det.linked_bev
        self.linked_extra_infos = mv_det.linked_extra_infos
        if self.obj_cls == 0: # Person
            self.linked_poses = mv_det.linked_poses
            self.linked_feats = mv_det.linked_feats
            self.linked_visses = mv_det.linked_visses
        
        self.estimate_bev_coor()
        self.last_bev_coor = self.bev_coor
        self.last_frame_idx = frameIdx

    def update(self, frame_idx): # Update the activate object
        
        self.hist_cam.append(self.linked_cam)
        self.hist_bbox.append(self.linked_bbox)
        self.hist_bev.append(self.linked_bev)
        if self.obj_cls == 0: # Person
            available_camera_idx = np.where(self.linked_cam != -1)[0]
            for aci in available_camera_idx:
                
                bbox_area = self.linked_bbox[aci][2] * self.linked_bbox[aci][3]
                dm_ratio = self.linked_extra_infos[aci][1] / bbox_area
                upper_body_pose_confs = np.mean(self.linked_poses[aci][:, 2][:7])

                if (dm_ratio > self.dm_ratio_thresh) or len(self.hist_feats) < 60:
                    self.hist_feats.append(self.linked_feats[aci])
                    self.hist_visses.append(self.linked_visses[aci])
                    self.hist_poses.append(self.linked_poses[aci])
                
                if (upper_body_pose_confs > self.upper_body_pose_thresh):
                    self.hist_heights.append(self.linked_extra_infos[aci][0])

        self.linked_cam = np.full(len(self.camkeys), -1)
        self.linked_bbox = np.zeros((len(self.camkeys), 4))
        self.linked_bev = np.zeros((len(self.camkeys), 4))
        if self.obj_cls == 0: # Person
            self.linked_poses = np.zeros((len(self.camkeys), 17, 3))
            self.linked_feats = np.zeros((len(self.camkeys), 6, 512))
            self.linked_visses = np.zeros((len(self.camkeys), 6))

        stt_flag = True

        if len(self.temp_dets):
            self.temp_dets = np.array(self.temp_dets).reshape(-1, 14)
            self.temp_feats = np.array(self.temp_feats).reshape(-1, 6, 512)
            self.temp_visses = np.array(self.temp_visses).reshape(-1, 6)

            for det, feat, viss, pose in zip(self.temp_dets, self.temp_feats, self.temp_visses, self.temp_poses):
                camIdx = int(det[0])
                self.linked_cam[camIdx] = int(det[2])
                self.linked_bbox[camIdx] = det[3:7]
                self.linked_bev[camIdx] = det[8:12]
                self.linked_extra_infos[camIdx] = det[12:14]
                if self.obj_cls == 0: # Person
                    self.linked_feats[camIdx] = feat
                    self.linked_visses[camIdx] = viss
                    self.linked_poses[camIdx] = pose
            self.estimate_bev_coor()
        else:
            stt_flag = False
        
        if self.bev_coor is None:
            stt_flag = False
        
        if stt_flag:
            self.last_bev_coor = self.bev_coor
            self.last_frame_idx = frame_idx

        self.temp_indices = []
        self.temp_dets = []
        self.temp_feats = []
        self.temp_visses = []

        return stt_flag


class Multiview_Others():

    def __init__(self, camkeys, dets, frame_idx = -1, history_len=360):

        self.camkeys = camkeys

        self.obj_cls = dets[0][1]   # Always the same class on the multiple det input

        self.linked_cam = np.full(len(camkeys), -1)
        self.linked_bbox = np.zeros((len(camkeys), 4), dtype=np.float32)
        self.linked_bev = np.zeros((len(camkeys), 4), dtype=np.float32)
        self.linked_extra_infos = np.zeros((len(camkeys), 2), dtype=np.float32)

        self.bev_coor = None

        self.obj_stt = -1
        self.global_id = -1

        self.last_bev_coor = None
        self.last_frame_idx = -1

        self.temp_indices = []
        self.temp_dets = []


        self.hist_cam = deque([], maxlen=history_len)
        self.hist_bbox = deque([], maxlen=history_len)
        self.hist_bev = deque([], maxlen=history_len)

        self.history_len = history_len
        self.temp_bev_coor_during_missing = deque([], maxlen=history_len)

        self.init_update(dets)
        self.estimate_bev_coor()

    def init_update(self, dets):
        for d_idx in range(dets.shape[0]):
            cam_d_idx = int(dets[d_idx][0])
            self.linked_cam[cam_d_idx] = dets[d_idx][2]
            self.linked_bev[cam_d_idx] = dets[d_idx][8:12]
            self.linked_bbox[cam_d_idx] = dets[d_idx][3:7]
            self.linked_extra_infos[cam_d_idx] = dets[d_idx][12:14]

    def estimate_bev_coor(self):
        available_bev_coors = self.linked_bev[self.linked_bev[:, 3] == 1][:, :3]
        if available_bev_coors.shape[0] > 0:
            self.bev_coor = available_bev_coors.mean(axis=0)
        else:
            self.bev_coor = None

    def update_using_single_det(self, det, frameIdx):
        d1_det_cam_idx = int(det[0])
        self.linked_cam[d1_det_cam_idx] = det[2]
        self.linked_bbox[d1_det_cam_idx] = det[3:7]
        self.linked_bev[d1_det_cam_idx] = det[8:12]
        self.linked_extra_infos[d1_det_cam_idx] = det[12:14]

        self.estimate_bev_coor()
        self.last_bev_coor = self.bev_coor
        self.last_frame_idx = frameIdx

    def update_using_mv_det(self, mv_det, frameIdx):
        self.linked_cam = mv_det.linked_cam
        self.linked_bbox = mv_det.linked_bbox
        self.linked_bev = mv_det.linked_bev
        self.linked_extra_infos = mv_det.linked_extra_infos

        self.estimate_bev_coor()
        self.last_bev_coor = self.bev_coor
        self.last_frame_idx = frameIdx

    def update(self, frame_idx): # Update the activate object
        self.hist_cam.append(self.linked_cam)
        self.hist_bbox.append(self.linked_bbox)
        self.hist_bev.append(self.linked_bev)

        self.linked_cam = np.full(len(self.camkeys), -1)
        self.linked_bbox = np.zeros((len(self.camkeys), 4))
        self.linked_bev = np.zeros((len(self.camkeys), 4))

        stt_flag = True
        if len(self.temp_dets):
            self.temp_dets = np.array(self.temp_dets).reshape(-1, 14)

            for det in self.temp_dets:
                camIdx = int(det[0])
                self.linked_cam[camIdx] = int(det[2])
                self.linked_bbox[camIdx] = det[3:7]
                self.linked_bev[camIdx] = det[8:12]
                self.linked_extra_infos[camIdx] = det[12:14]
            self.estimate_bev_coor()
        else:
            stt_flag = False

        if self.bev_coor is None:
            stt_flag = False

        if stt_flag:
            self.last_bev_coor = self.bev_coor
            self.last_frame_idx = frame_idx

        self.temp_indices = []
        self.temp_dets = []

        return stt_flag


class MTrack():
    def __init__(self, camkeys, create_new_objs=True):

        self.ID_generator = ID_Generator()
        self.camera_keys = camkeys

        # Person only
        self.tracked_objs = []
        self.missing_objs = []

        self.bev_dist_thresh = 1.0
        self.reid_thresh = 0.35

        self.max_dist_per_second = 2.0
        self.fps = 30
        self.reid_thresh_for_revoke_missing = 0.3

        self.close_global_thresh = 0.5


        # Others
        # Person, ForkLift, Nova, Trans, Fourier, Agi
        self.tracked_others = []
        self.missing_others = []
        self.bev_dist_thresh_others = [
            0, 5, 2, 2, 1.5, 1
        ]

        self.create_new_objs = create_new_objs

        self.cur_frameIdx = 0

    # For Person Class
    def group_person(self, dets, poses, feats, visses):
        # First match using bev + feats
        person_indices = dets[:, 1] == 0
        person_dets = dets[person_indices]
        person_poses = poses[person_indices]
        person_feats = feats[person_indices]
        person_visses = visses[person_indices]
        person_group_idx = pita_utils.clustering_person(person_dets, person_poses, person_feats, person_visses, 
                                    self.bev_dist_thresh, self.reid_thresh)

        unique_person_group_idx = np.unique(person_group_idx)
        mv_persons = []
        for upg_idx in unique_person_group_idx:
            if upg_idx >= 0:
                upg_idx_person_dets = person_dets[person_group_idx==upg_idx]
                upg_idx_person_poses = person_poses[person_group_idx==upg_idx]
                upg_idx_person_feats = person_feats[person_group_idx==upg_idx]
                upg_idx_person_visses = person_visses[person_group_idx==upg_idx]
                mv_persons.append(Multiview_Person(self.camera_keys, upg_idx_person_dets, upg_idx_person_poses, 
                                        upg_idx_person_feats, upg_idx_person_visses))
        
        # Second match only use bev (more strict for bev threshold)
        remain_person_dets = person_dets[np.where(person_group_idx == -2)[0]]
        remain_person_poses = person_poses[np.where(person_group_idx == -2)[0]]
        remain_person_feats = person_feats[np.where(person_group_idx == -2)[0]]
        remain_person_visses = person_visses[np.where(person_group_idx == -2)[0]]
        if len(remain_person_dets) and len(mv_persons):
            r_person_group_idx = pita_utils.clustering_dets_bev_only(remain_person_dets, mv_dets=mv_persons,
                                            bev_dist_thresh=self.bev_dist_thresh / 2)
            # print(r_person_group_idx)
            unique_r_person_group_idx = np.unique(r_person_group_idx)
            for upg_idx in unique_r_person_group_idx:
                if upg_idx >= 0:
                    upg_idx_person_dets = remain_person_dets[r_person_group_idx==upg_idx]
                    upg_idx_person_poses = remain_person_poses[r_person_group_idx==upg_idx]
                    upg_idx_person_feats = remain_person_feats[r_person_group_idx==upg_idx]
                    upg_idx_person_visses = remain_person_visses[r_person_group_idx==upg_idx]
                    mv_persons.append(Multiview_Person(self.camera_keys, upg_idx_person_dets, upg_idx_person_poses, 
                                            upg_idx_person_feats, upg_idx_person_visses))

        return mv_persons

    def confirm_person_update(self):
        return_det_idx_list = []
        for tobj_idx in range(len(self.tracked_objs)):
            track_obj = self.tracked_objs[tobj_idx]
            link_cam_list = np.where(track_obj.linked_cam != -1)[0]
            
            if len(track_obj.temp_dets) > 1:

                if len(track_obj.hist_feats) == 0:
                    gallery_feats, gallery_visses = track_obj.return_available_feats_visses()
                else:
                    gallery_feats = np.array(track_obj.hist_feats)
                    gallery_visses = np.array(track_obj.hist_visses)

                temp_feats = np.array(track_obj.temp_feats).reshape(-1, 6, 512)
                temp_visses = np.array(track_obj.temp_visses).reshape(-1, 6)

                reid_dist = pita_utils.estimate_embedding_distance_kpr_2(
                    temp_feats, gallery_feats, temp_visses, gallery_visses
                )

                # reid_dist_min = np.min(reid_dist, axis=1) # Pikachu
                reid_dist_min = np.mean(reid_dist, axis=1)

                not_fit_det_indices = np.where(reid_dist_min > self.reid_thresh)[0]
              
                temp_bev_dist_mat = pita_utils.estimate_bev_matrix_dist(np.array(track_obj.temp_dets))
                # if len(track_obj.temp_dets) > 1:
                bev_dist_temp_list = []
                for i_temp in range(temp_bev_dist_mat.shape[0]):
                    not_fit_flag = False
                    i_temp_bev_dist = temp_bev_dist_mat[i_temp].sum() / (len(temp_bev_dist_mat) - 1)
                    bev_dist_temp_list.append(i_temp_bev_dist)
                    if (i_temp_bev_dist > self.bev_dist_thresh):
                        not_fit_flag = True
                        # not_fit_det_indices = np.append(not_fit_det_indices, i_temp)

                    # i_temp_pose_upper_confs = np.mean(track_obj.temp_poses[i_temp][:, 2][:7])
                    # i_temp_heigh = track_obj.temp_dets[i_temp][-2]
                    # mean_hist_height = np.mean(track_obj.hist_heights)
                    # print(mean_hist_height)
                    # if i_temp_pose_upper_confs > track_obj.upper_body_pose_thresh:
                    #     if np.abs(i_temp_heigh - mean_hist_height) > 0.05:
                    #         not_fit_flag = True
                    if not_fit_flag:
                        not_fit_det_indices = np.append(not_fit_det_indices, i_temp)
                # else:
                #     if temp_bev_dist_mat[0][0] > self.bev_dist_thresh / 2:
                #         not_fit_det_indices = np.append(not_fit_det_indices, 0)
                return_det_idx_list.append(not_fit_det_indices)
            else:
                return_det_idx_list.append([])

            
        return return_det_idx_list

    def match_new_mv_persons_to_tracked_person(self, person_dets, person_poses, person_feats, person_visses):
        remain_dets = np.ones(len(person_dets))
        for tobj_idx in range(len(self.tracked_objs)):
            tracked_obj = self.tracked_objs[tobj_idx]
            link_cam_list = np.where(tracked_obj.linked_cam != -1)[0]
            link_cam_track_objID_list = tracked_obj.linked_cam[link_cam_list]

            for camIdx, trackID in zip(link_cam_list, link_cam_track_objID_list):
                match_det_idx = np.where( (person_dets[:, 0] == camIdx) & (person_dets[:, 2] == trackID))[0]
                if len(match_det_idx) == 1:
                    match_det_idx = match_det_idx[0]
                    remain_dets[match_det_idx] = 0
                    self.tracked_objs[tobj_idx].temp_indices.append(match_det_idx)
                    self.tracked_objs[tobj_idx].temp_dets.append(person_dets[match_det_idx])
                    self.tracked_objs[tobj_idx].temp_feats.append(person_feats[match_det_idx])
                    self.tracked_objs[tobj_idx].temp_visses.append(person_visses[match_det_idx])
                    self.tracked_objs[tobj_idx].temp_poses.append(person_poses[match_det_idx])
                elif len(match_det_idx) > 1:
                    print('What the problem')

        not_match_dets_list = self.confirm_person_update()
        for tobj_idx in range(len(self.tracked_objs)):
            tobj_idx_not_match_temp = sorted(np.unique(not_match_dets_list[tobj_idx]), reverse=True)
            for temp_idx in tobj_idx_not_match_temp:
                back_remain_idx = self.tracked_objs[tobj_idx].temp_indices.pop(temp_idx)
                del self.tracked_objs[tobj_idx].temp_dets[temp_idx]
                del self.tracked_objs[tobj_idx].temp_feats[temp_idx]
                del self.tracked_objs[tobj_idx].temp_visses[temp_idx]
                del self.tracked_objs[tobj_idx].temp_poses[temp_idx]
                remain_dets[back_remain_idx] = 1
        
        missing_obj_indices = []
        for tobj_idx in range(len(self.tracked_objs)):
            obj_stt = self.tracked_objs[tobj_idx].update(self.cur_frameIdx)
            if obj_stt == False:
                missing_obj_indices.append(tobj_idx)

        missing_obj_indices = sorted(missing_obj_indices, reverse=True)
        for mobj_idx in missing_obj_indices:
            missing_obj = self.tracked_objs.pop(mobj_idx) # Remove the missing tracked objects
            self.missing_objs.append(missing_obj)

        remain_person_indices = np.where(remain_dets==1)[0]

        return person_dets[remain_person_indices], person_poses[remain_person_indices], person_feats[remain_person_indices], person_visses[remain_person_indices]

    def match_remain_person_to_tracked_person(self, dets_1, poses_1, feats_1, visses_1):

        if len(dets_1):
            # Object need to be close on BEV
            tracked_objs_bev_coor_arr = []
            for tobj in range(len(self.tracked_objs)):
                tracked_objs_bev_coor_arr.append(self.tracked_objs[tobj].bev_coor)
            tracked_objs_bev_coor_arr = np.array(tracked_objs_bev_coor_arr)
            
            bev_dist_mat = cdist(dets_1[:, 8:11], tracked_objs_bev_coor_arr)

            # Make sure no match the same camera
            for d1_idx in range(len(dets_1)):
                d1_det_cam_idx = int(dets_1[d1_idx][0])
                for tobj in range(len(self.tracked_objs)):
                    if self.tracked_objs[tobj].linked_cam[d1_det_cam_idx] >= 0:
                        bev_dist_mat[d1_idx][tobj] = np.inf

            remain_person_dets_1 = np.ones(len(dets_1))
            for d1_idx in range(len(dets_1)):
                bev_close_indices = np.where(bev_dist_mat[d1_idx] < self.bev_dist_thresh)[0]

                if len(bev_close_indices):
                    d1_idx_feats = feats_1[d1_idx].reshape(1, 6, 512)
                    d1_idx_visses = visses_1[d1_idx].reshape(1, 6)
                    stacked_reid_dist = np.full(len(bev_close_indices), np.inf)
                    for idx_bev_close_indices in range(len(bev_close_indices)):
                        bev_close_indice = bev_close_indices[idx_bev_close_indices]
                        # gallery_feats, gallery_visses = self.tracked_objs[bev_close_indice].return_filtered_feats_visses(self.filter_viss_sample_step)
                        gallery_feats = np.array(self.tracked_objs[bev_close_indice].hist_feats)
                        gallery_visses = np.array(self.tracked_objs[bev_close_indice].hist_visses)
                        reid_dist = pita_utils.estimate_embedding_distance_kpr_2(d1_idx_feats, gallery_feats, d1_idx_visses, gallery_visses)
                        stacked_reid_dist[idx_bev_close_indices] = np.min(reid_dist[0])

                    # The local obj is too close to other global objects
                    # It need to be ignored and removed for the later stage 
                    # in which we refind the missing people and possible generate new global objects
                    remain_person_dets_1[d1_idx] = 0 
                    if stacked_reid_dist[np.argmin(stacked_reid_dist)] < self.reid_thresh:
                        # Update to the tracked obj directly
                        map_tracked_obj_idx = bev_close_indices[np.argmin(stacked_reid_dist)]
                        self.tracked_objs[map_tracked_obj_idx].update_using_single_det(dets_1[d1_idx], poses_1[d1_idx], feats_1[d1_idx], visses_1[d1_idx], self.cur_frameIdx)

            # print("Remained person dets: ", len(np.where(remain_person_dets_1==1)[0]))
            # print("No BEV person dets: ", len(np.where(dets[:, 11]==0)[0]))
            remain_indices = np.where(remain_person_dets_1==1)[0]
            remain_dets = dets_1[remain_indices]
            remain_poses = poses_1[remain_indices]
            remain_feats = feats_1[remain_indices]
            remain_visses = visses_1[remain_indices]

            return remain_dets, remain_poses, remain_feats, remain_visses

        else:
            return [], [], [], []

    def refind_missing_person(self, dets, poses, feats, visses):

        if len(self.missing_objs) > 0 and len(dets) > 0:

            remain_person = np.ones(len(dets))

            mean_reid_dist_list = np.full((len(dets), len(self.missing_objs)), np.inf)
            for d_idx in range(len(dets)):
                d_idx_feat = feats[d_idx].reshape(1, 6, 512)
                d_idx_viss = visses[d_idx].reshape(1, 6)
                for mobj_idx in range(len(self.missing_objs)):
                    mobj_feats = np.array(self.missing_objs[mobj_idx].hist_feats)
                    mobj_visses = np.array(self.missing_objs[mobj_idx].hist_visses)
                    reid_dist = pita_utils.estimate_embedding_distance_kpr_2(d_idx_feat, mobj_feats, d_idx_viss, mobj_visses)

                    mean_reid_dist_list[d_idx][mobj_idx] = np.mean(reid_dist[0])
            
            dets_missing_obj_map_list = np.full(len(dets), -1)
            conflict_det_indices = []

            for d_idx in range(len(dets)):
                linked_missing_obj_list = np.where(mean_reid_dist_list[d_idx] < self.reid_thresh)[0]
                if len(linked_missing_obj_list) > 0:
                    if len(linked_missing_obj_list) > 1:
                        conflict_det_indices.append(d_idx)
                    else:
                        dets_missing_obj_map_list[d_idx] = linked_missing_obj_list[0]
                    
            potential_missing_obj_list = np.unique(dets_missing_obj_map_list)
            revoke_missing_obj_list = []
            for potential_mobj_idx in potential_missing_obj_list:
                if potential_mobj_idx != -1:
                    possible_related_dets = np.where(dets_missing_obj_map_list == potential_mobj_idx)[0]
                    number_of_possible_related_dets = len(possible_related_dets)
                    if number_of_possible_related_dets > 0:
                        potential_dets = dets[possible_related_dets]
                        potential_dets_flag = True
                        if number_of_possible_related_dets > 1:
                            if not pita_utils.check_dets_not_from_same_cam_and_close_to_each_other(potential_dets, self.bev_dist_thresh):
                                potential_dets_flag = False
                        if potential_dets_flag: # Revoke missing object
                            revoke_missing_obj_list.append(potential_mobj_idx)
                            for p_det_idx in possible_related_dets:
                                remain_person[p_det_idx] = 0
                                self.missing_objs[potential_mobj_idx].update_using_single_det(
                                    dets[p_det_idx], poses[p_det_idx], feats[p_det_idx], visses[p_det_idx], self.cur_frameIdx
                                )
            revoke_missing_obj_list = sorted(revoke_missing_obj_list, reverse=True)


            for r_mobj_idx in revoke_missing_obj_list:
                r_mobj = self.missing_objs.pop(r_mobj_idx)
                self.tracked_objs.append(r_mobj)

            reman_person_indices = np.where(remain_person == 1)[0]                  
            return dets[reman_person_indices], poses[reman_person_indices], feats[reman_person_indices], visses[reman_person_indices]
        else:
            if len(dets):
                return dets, poses, feats, visses
            else:
                return [], [], [], []

    def refind_missing_person_v3(self, dets, poses, feats, visses):

        if len(self.missing_objs) > 0 and len(dets) > 0:

            # print('Missing person: ')
            # for mperson in self.missing_objs:
            #     print(mperson.global_id, end=' ')
            # print('')

            remain_person = np.ones(len(dets))
            mean_reid_dist_list = np.full((len(dets), len(self.missing_objs)), np.inf)
            for d_idx in range(len(dets)):
                d_idx_feat = feats[d_idx].reshape(1, 6, 512)
                d_idx_viss = visses[d_idx].reshape(1, 6)
                for mobj_idx in range(len(self.missing_objs)):
                    mobj_feats = np.array(self.missing_objs[mobj_idx].hist_feats)
                    mobj_visses = np.array(self.missing_objs[mobj_idx].hist_visses)
                    reid_dist = pita_utils.estimate_embedding_distance_kpr_2(d_idx_feat, mobj_feats, d_idx_viss, mobj_visses)

                    mean_reid_dist_list[d_idx][mobj_idx] = np.mean(reid_dist[0])
            
            dets_missing_obj_map_list = np.full(len(dets), -1)
            conflict_det_indices = []

            for d_idx in range(len(dets)):
                linked_missing_obj_list = np.where(mean_reid_dist_list[d_idx] < self.reid_thresh)[0]
                if len(linked_missing_obj_list) > 0:
                    if len(linked_missing_obj_list) > 1:
                        conflict_det_indices.append(d_idx)
                    else:
                        dets_missing_obj_map_list[d_idx] = linked_missing_obj_list[0]

            # Add more to conflct det indices using bev distance control
            bev_able_map = np.zeros((len(dets), len(self.missing_objs)))
            for d_idx in range(len(dets)):
                for mobj_idx in range(len(self.missing_objs)):
                    x_dist = dets[d_idx][8] - self.missing_objs[mobj_idx].last_bev_coor[0]
                    y_dist = dets[d_idx][9] - self.missing_objs[mobj_idx].last_bev_coor[1]
                    xy_dist = np.sqrt(x_dist**2 + y_dist**2)
                    bev_limit = (self.cur_frameIdx - self.missing_objs[mobj_idx].last_frame_idx) / (self.fps / self.max_dist_per_second)
                    if xy_dist < bev_limit:
                        bev_able_map[d_idx][mobj_idx] = 1

                    
            potential_missing_obj_list = np.unique(dets_missing_obj_map_list)
            # print('potential_missing_obj_list: ', potential_missing_obj_list)
            revoke_missing_obj_list = []
            for potential_mobj_idx in potential_missing_obj_list:
                if potential_mobj_idx != -1:
                    possible_related_dets = np.where(dets_missing_obj_map_list == potential_mobj_idx)[0]
                    number_of_possible_related_dets = len(possible_related_dets)
                    if number_of_possible_related_dets > 0:
                        potential_dets = dets[possible_related_dets]
                        potential_dets_flag = True
                        for p_det_idx in possible_related_dets:
                            if bev_able_map[p_det_idx][potential_mobj_idx] == 0:
                                potential_dets_flag = False
                        if number_of_possible_related_dets > 1:
                            if not pita_utils.check_dets_not_from_same_cam_and_close_to_each_other(potential_dets, self.bev_dist_thresh):
                                potential_dets_flag = False
                        if potential_dets_flag: # Revoke missing object
                            revoke_missing_obj_list.append(potential_mobj_idx)
                            for p_det_idx in possible_related_dets:
                                remain_person[p_det_idx] = 0
                                self.missing_objs[potential_mobj_idx].update_using_single_det(
                                    dets[p_det_idx], poses[p_det_idx], feats[p_det_idx], visses[p_det_idx], self.cur_frameIdx
                                )
            revoke_missing_obj_list = sorted(revoke_missing_obj_list, reverse=True)


            for r_mobj_idx in revoke_missing_obj_list:
                r_mobj = self.missing_objs.pop(r_mobj_idx)
                r_mobj.temp_bev_coor_during_missing = deque([], maxlen=r_mobj.history_len)
                self.tracked_objs.append(r_mobj)

            remain_person_indices = np.where(remain_person == 1)[0]                  
            return dets[remain_person_indices], poses[remain_person_indices], feats[remain_person_indices], visses[remain_person_indices]
        else:
            if len(dets):
                return dets, poses, feats, visses
            else:
                return [], [], [], []

    def refind_missing_person_v2(self, dets, poses, feats, visses):

        if len(self.missing_objs) > 0 and len(dets) > 0:
            remain_person = np.ones(len(dets))

            person_group_idx = pita_utils.clustering_person(dets, poses, feats, visses, self.bev_dist_thresh, self.reid_thresh)
            unique_person_group_indices = np.unique(person_group_idx)

            mv_persons = []
            mv_person_bev_arr = []
            mv_person_feat_arr = []
            mv_person_viss_arr = []
            for upg_idx in unique_person_group_indices:
                if upg_idx >= 0:
                    upg_idx_person_dets = dets[person_group_idx==upg_idx]
                    upg_idx_person_poses = poses[person_group_idx==upg_idx]
                    upg_idx_person_feats = feats[person_group_idx==upg_idx]
                    upg_idx_person_visses = visses[person_group_idx==upg_idx]
                    mv_persons.append(Multiview_Person(self.camera_keys, upg_idx_person_dets, upg_idx_person_poses, 
                                            upg_idx_person_feats, upg_idx_person_visses))
            
            if len(mv_persons):

                for mvp_idx in range(len(mv_persons)):
                    mvp_feats, mvp_visses = mv_persons[mvp_idx].return_available_feats_visses()
                    mv_person_bev_arr.append(mv_persons[mvp_idx].bev_coor)
                    mv_person_feat_arr.append(mvp_feats)
                    mv_person_viss_arr.append(mvp_visses)
                mv_person_bev_arr = np.array(mv_person_bev_arr)

                mobj_bev_arr = []
                mobj_last_frame = []
                mobj_feats_arr = []
                mobj_visses_arr = []
                mobj_limit_dist_arr = np.zeros(len(self.missing_objs))

                for mobj_idx in range(len(self.missing_objs)):
                    mobj_bev_arr.append(self.missing_objs[mobj_idx].last_bev_coor)
                    mobj_last_frame.append(self.missing_objs[mobj_idx].last_frame_idx)
                    mobj_feats = np.array(self.missing_objs[mobj_idx].hist_feats)
                    mobj_visses = np.array(self.missing_objs[mobj_idx].hist_visses)
                    mobj_feats_arr.append(mobj_feats)
                    mobj_visses_arr.append(mobj_visses)
                    mobj_limit_dist_arr[mobj_idx] = (self.cur_frameIdx - self.missing_objs[mobj_idx].last_frame_idx) / (self.fps / self.max_dist_per_second)
                mobj_bev_arr = np.array(mobj_bev_arr)

                bev_dist_map = cdist(mobj_bev_arr, mv_person_bev_arr)

                reid_dist_map = np.ones((len(self.missing_objs), len(mv_persons)))
                for mobj_idx in range(len(self.missing_objs)):
                    for mvp_idx in range(len(mv_persons)):
                        m_d_reid_dist = pita_utils.estimate_embedding_distance_kpr_2(
                            mobj_feats_arr[mobj_idx], mv_person_feat_arr[mvp_idx], 
                            mobj_visses_arr[mobj_idx], mv_person_viss_arr[mvp_idx]
                            )
                        # reid_dist_map[mobj_idx][mvp_idx] = np.mean(m_d_reid_dist) * 2
                        reid_dist_map[mobj_idx][mvp_idx] = np.min(m_d_reid_dist)


                revoke_missing_list = np.full(len(self.missing_objs), -1)
                revoke_missing_idx_list = []

                mobj_match_list = []
                mobj_match_bev_only_list = []
                for mobj_idx in range(len(self.missing_objs)):
                    bev_match_list = bev_dist_map[mobj_idx] < mobj_limit_dist_arr[mobj_idx]
                    reid_match_list = reid_dist_map[mobj_idx] < self.reid_thresh_for_revoke_missing
                    # print(self.missing_objs[mobj_idx].global_id, bev_match_list, reid_match_list)
                    # print(self.missing_objs[mobj_idx].global_id, mobj_limit_dist_arr[mobj_idx])
                    # print(np.where(bev_match_list & reid_match_list)[0])
                    mobj_match_list.append(np.where(bev_match_list & reid_match_list)[0])
                    mobj_match_bev_only_list.append(np.where(bev_match_list)[0])
                    # print(self.missing_objs[mobj_idx].global_id, mobj_match_list[mobj_idx])

                unique_match_list = list(set(tuple(x) for x in mobj_match_list))
                for mobj_idx in range(len(self.missing_objs)):
                    if len(mobj_match_list[mobj_idx]) == 1: # match exactly 1 target
                        if mobj_match_list[mobj_idx] in unique_match_list:
                            revoke_missing_list[mobj_idx] = mobj_match_list[mobj_idx][0]
                            revoke_missing_idx_list.append(mobj_idx)
                    # else:
                    #     continue # Either 0 or >1, which is not what we want for a reactive cases

                # unique_match_bev_only_list = list(set(tuple(x) for x in mobj_match_bev_only_list))
                # for mobj_idx in range(len(self.missing_objs)):
                #     if len(mobj_match_bev_only_list[mobj_idx]) == 1:
                #         if mobj_match_bev_only_list[mobj_idx] in unique_match_bev_only_list:
                #             revoke_missing_list[mobj_idx] = mobj_match_bev_only_list[mobj_idx][0]
                #             revoke_missing_idx_list.append(mobj_idx)

                for mobj_idx in range(len(self.missing_objs)):
                    if revoke_missing_list[mobj_idx] != -1:
                        self.missing_objs[mobj_idx].update_using_mv_det(
                                mv_persons[revoke_missing_list[mobj_idx]], self.cur_frameIdx)

                revoke_missing_idx_list = sorted(np.unique(revoke_missing_idx_list), reverse=True)
                # print(revoke_missing_idx_list)
                for r_mobj_idx in revoke_missing_idx_list:
                    r_mobj = self.missing_objs.pop(r_mobj_idx)
                    self.tracked_objs.append(r_mobj)


        else:
            if len(dets):
                return dets, poses, feats, visses
            else:
                return [], [], [], []

    def create_new_persons(self, dets, poses, feats, visses):
        if (len(dets)) and len(self.missing_objs) == 0:
            mv_persons = self.group_person(dets, poses, feats, visses)

            bev_mv_person_check_list = np.ones(len(mv_persons))
            # Person
            # Check BeV Dist -> Need to be far away from every tracked objects (3 times the normal bev_dist_thresh)
            for mvp_idx in range(len(mv_persons)):
                mv_person = mv_persons[mvp_idx]
                for tobj in self.tracked_objs:
                    tobj_bev = tobj.bev_coor
                    x_dist = mv_person.bev_coor[0] - tobj_bev[0]
                    y_dist = mv_person.bev_coor[1] - tobj_bev[1]
                    bev_dist = np.sqrt(x_dist**2 + y_dist**2)
                    if bev_dist < (self.bev_dist_thresh * 3):
                        bev_mv_person_check_list[mvp_idx] = 0
                
            
            for mvp_idx in range(len(mv_persons)):
                if bev_mv_person_check_list[mvp_idx] == 1:
                    self.tracked_objs.append(mv_persons[mvp_idx])
                    self.tracked_objs[-1].global_id = self.ID_generator.get_cur_id()

    # For other classes
    def return_others_by_class_id(self, obj_cls, tracked_flag=True):
        return_objs = []
        if tracked_flag:
            for obj in self.tracked_others:
                if obj.obj_cls == obj_cls:
                    return_objs.append(obj)
        else:
            for obj in self.missing_others:
                if obj.obj_cls == obj_cls:
                    return_objs.append(obj)
        return return_objs

    def group_others(self, dets):
        others_indices = dets[:, 1] != 0
        other_dets = dets[others_indices]
        other_group_idx = pita_utils.clustering_dets_bev_only(other_dets, bev_dist_thresh=self.bev_dist_thresh)
        unique_other_group_idx = np.unique(other_group_idx)
        mv_others = []
        for uog_idx in unique_other_group_idx:
            if uog_idx >= 0:
                uog_idx_other_dets = other_dets[other_group_idx==uog_idx]
                mv_others.append(Multiview_Others(self.camera_keys, uog_idx_other_dets))
        return mv_others

    def confirm_others_update(self):
        return_det_idx_list = []
        for tobj_idx in range(len(self.tracked_others)):
            track_obj = self.tracked_others[tobj_idx]

            not_fit_det_indices = []
            if len(track_obj.temp_dets) > 1:
                temp_bev_dist_mat = pita_utils.estimate_bev_matrix_dist(np.array(track_obj.temp_dets))
                bev_dist_temp_list = []
                for i_temp in range(temp_bev_dist_mat.shape[0]):
                    i_temp_bev_dist = temp_bev_dist_mat[i_temp].sum() / (len(temp_bev_dist_mat) - 1)
                    bev_dist_temp_list.append(i_temp_bev_dist)
                    if (i_temp_bev_dist > self.bev_dist_thresh_others[int(track_obj.obj_cls)]):
                        not_fit_det_indices = np.append(not_fit_det_indices, i_temp)
                return_det_idx_list.append(not_fit_det_indices)
            else:
                return_det_idx_list.append([])
        return return_det_idx_list

    def match_new_mv_other_to_tracked_others(self, dets):
        remain_dets = np.ones(len(dets))

        for tobj_idx in range(len(self.tracked_others)):
            tracked_obj = self.tracked_others[tobj_idx]
            link_cam_list = np.where(tracked_obj.linked_cam != -1)[0]
            link_cam_track_objID_list = tracked_obj.linked_cam[link_cam_list]

            for camIdx, trackID in zip(link_cam_list, link_cam_track_objID_list):
                match_det_idx = np.where( (dets[:, 0] == camIdx) & (dets[:, 2] == trackID))[0]
                if len(match_det_idx) == 1:
                    match_det_idx = match_det_idx[0]
                    remain_dets[match_det_idx] = 0
                    self.tracked_others[tobj_idx].temp_indices.append(match_det_idx)
                    self.tracked_others[tobj_idx].temp_dets.append(dets[match_det_idx])
                elif len(match_det_idx) > 1:
                    print('What the problem')


        not_match_dets_list = self.confirm_others_update()
        # print('Not match dets list: ', not_match_dets_list)

        for tobj_idx in range(len(self.tracked_others)):
            tobj_idx_not_match_temp = sorted(np.unique(not_match_dets_list[tobj_idx]), reverse=True)
            for temp_idx in tobj_idx_not_match_temp:
                back_remain_idx = self.tracked_others[tobj_idx].temp_indices.pop(int(temp_idx))
                del self.tracked_others[tobj_idx].temp_dets[int(temp_idx)]
                remain_dets[back_remain_idx] = 1


        missing_obj_indices = []
        for tobj_idx in range(len(self.tracked_others)):
            obj_stt = self.tracked_others[tobj_idx].update(self.cur_frameIdx)
            if obj_stt == False:
                missing_obj_indices.append(tobj_idx)

        missing_obj_indices = sorted(missing_obj_indices, reverse=True)
        for mobj_idx in missing_obj_indices:
            missing_obj = self.tracked_others.pop(mobj_idx) # Remove the missing tracked objects
            missing_obj.temp_bev_coor_during_missing = deque([], maxlen=missing_obj.history_len)
            self.missing_others.append(missing_obj)

        remain_other_indices = np.where(remain_dets==1)[0]
        return dets[remain_other_indices]

    def match_remain_other_to_tracked_others(self, dets):
        if len(dets):
            # Object need to be close on BEV
            tracked_objs_bev_coor_arr = []
            for tobj in range(len(self.tracked_others)):
                tracked_objs_bev_coor_arr.append(self.tracked_others[tobj].bev_coor)
            tracked_objs_bev_coor_arr = np.array(tracked_objs_bev_coor_arr)

            bev_dist_mat = cdist(dets[:, 8:11], tracked_objs_bev_coor_arr)

            # Make sure no match the same camera and no match on wrong class
            for d_idx in range(len(dets)):
                d_det_cam_idx = int(dets[d_idx][0])
                for tobj in range(len(self.tracked_others)):
                    if self.tracked_others[tobj].linked_cam[d_det_cam_idx] >= 0:
                        bev_dist_mat[d_idx][tobj] = np.inf
                    if self.tracked_others[tobj].obj_cls != dets[d_idx][1]:
                        bev_dist_mat[d_idx][tobj] = np.inf

            remain_other_dets = np.ones(len(dets))
            for d_idx in range(len(dets)):
                closest_bev_idx = np.argmin(bev_dist_mat[d_idx])
                if bev_dist_mat[d_idx][closest_bev_idx] < self.bev_dist_thresh_others[int(dets[d_idx][1])]:
                    remain_other_dets[d_idx] = 0
                    self.tracked_others[closest_bev_idx].update_using_single_det(dets[d_idx], self.cur_frameIdx)

            remain_indices = np.where(remain_other_dets==1)[0]
            return dets[remain_indices]

        return dets

    def refind_missing_others(self, dets):
        if len(self.missing_others) > 0 and len(dets) > 0:

            mv_others = self.group_others(dets)
            
            if len(mv_others):
                mv_others_match_list = np.full(len(mv_others), -1)
                mv_others_dist_list = np.full(len(mv_others), np.inf)

                for check_obj_cls in range(1,6,1):
                    c_mobj_index_list = []
                    # Get all missing others according to their class
                    for mobj_idx in range(len(self.missing_others)):
                        mobj = self.missing_others[mobj_idx]
                        if mobj.obj_cls == check_obj_cls:
                            c_mobj_index_list.append(mobj_idx)

                    c_mv_index_list = []
                    for mvo_idx in range(len(mv_others)):
                        mv_other = mv_others[mvo_idx]
                        if mv_other.obj_cls == check_obj_cls:
                            c_mv_index_list.append(mvo_idx)

                    for cmoi in c_mobj_index_list:
                        for cmvi in c_mv_index_list:
                            bev_limit = (self.cur_frameIdx - self.missing_others[cmoi].last_frame_idx) / (self.fps / self.max_dist_per_second)
                            bev_dist = pita_utils.bev_dist_between_2_mv_objs(self.missing_others[cmoi], mv_others[cmvi])
                            if bev_dist < bev_limit:
                                if mv_others_match_list[cmvi] == -1:
                                    mv_others_match_list[cmvi] = cmoi
                                    mv_others_dist_list[cmvi] = bev_dist
                                else:
                                    if bev_dist < mv_others_dist_list[cmvi]:
                                        mv_others_match_list[cmvi] = cmoi
                                        mv_others_dist_list[cmvi] = bev_dist

                for mvo_idx in range(len(mv_others)):
                    match_mobj_idx = mv_others_match_list[mvo_idx]
                    if match_mobj_idx != -1:
                        same_potential_matches = np.where(mv_others_match_list == match_mobj_idx)[0]
                        if len(same_potential_matches) > 1:
                            dist_list = mv_others_dist_list[same_potential_matches]
                            keep_mvo_idx = np.argmin(dist_list)
                            if keep_mvo_idx != mvo_idx:
                                mv_others_match_list[mvo_idx] = -1

                revoke_other_list = []
                for mvo_idx in range(len(mv_others)):
                    match_mobj_idx = mv_others_match_list[mvo_idx]
                    if match_mobj_idx != -1:  
                        revoke_other_list.append(match_mobj_idx)
                        self.missing_others[match_mobj_idx].update_using_mv_det(mv_others[mvo_idx], self.cur_frameIdx)

                revoke_other_list = sorted(revoke_other_list, reverse=True)
                for r_mobj_idx in revoke_other_list:
                    r_mobj = self.missing_others.pop(r_mobj_idx)
                    r_mobj.temp_bev_coor_during_missing = deque([], maxlen=r_mobj.history_len)
                    self.tracked_others.append(r_mobj)

                remain_mv_other_indices = np.where(mv_others_match_list == -1)[0]
                if len(remain_mv_other_indices):
                    return_mv_others = []
                    for r_mv_idx in remain_mv_other_indices:
                        return_mv_others.append(mv_others[int(r_mv_idx)])
                    return return_mv_others
                else:
                    return []
            return []
        else:
            return []
        #         return 

        #     else:
        #         return [], dets
            
        # else:
        #     if len(dets):
        #         return [], dets
        #     else:
        #         return [], []

    def make_too_close_tracked_missing_together(self):
        make_missing_list = []

        tracked_objs_bev_coor_arr = []
        for tobj in range(len(self.tracked_objs)):
            tracked_objs_bev_coor_arr.append(self.tracked_objs[tobj].bev_coor)
        tracked_objs_bev_coor_arr = np.array(tracked_objs_bev_coor_arr)

        bev_dist_mat = cdist(tracked_objs_bev_coor_arr, tracked_objs_bev_coor_arr)
        too_close_objs = bev_dist_mat < self.close_global_thresh

        for tobj in range(len(self.tracked_objs)):
            tobj_too_close_objs = np.where(too_close_objs[tobj])[0]
            tobj_too_close_objs = [x for x in tobj_too_close_objs if x != tobj]
            make_missing_list.append(tobj_too_close_objs)

        # print('bev_dist_mat')
        # print(make_missing_list)

        # for idx in range(len(make_missing_list)):
        #     print(self.tracked_objs[idx].global_id, end=' ')
        #     for idx_j in make_missing_list[idx]:
        #         print(self.tracked_objs[idx_j].global_id, end=' ')
        #     print('')

        make_missing_list = [item for sublist in make_missing_list for item in sublist]
        make_missing_list = np.unique(make_missing_list)
        make_missing_list = sorted(make_missing_list, reverse=True)

        for item in make_missing_list:
            make_miss_obj = self.tracked_objs.pop(item)
            self.missing_objs.append(make_miss_obj)

    # For postprocessing    
    def find_the_temp_bev_for_missing_person(self, dets, poses, feats, visses):

        if len(self.missing_objs) and len(dets): # Only work when there are more than one missing person

            mv_persons = self.group_person(dets, poses, feats, visses)
            if len(mv_persons):
                mv_person_bev_coor_list = []
                for mv_p_idx in range(len(mv_persons)):
                    mv_person_bev_coor_list.append(mv_persons[mv_p_idx].bev_coor)
                mv_person_bev_coor_arr = np.array(mv_person_bev_coor_list)

                missing_person_bev_coor_list = []
                for mp_idx in range(len(self.missing_objs)):
                    missing_person = self.missing_objs[mp_idx]
                    if len(missing_person.temp_bev_coor_during_missing):
                        missing_person_bev_coor_list.append(missing_person.temp_bev_coor_during_missing[-1])
                    else:
                        missing_person_bev_coor_list.append(missing_person.last_bev_coor)
                        missing_person.temp_bev_coor_during_missing.append(missing_person.last_bev_coor)
                missing_person_bev_coor_arr = np.array(missing_person_bev_coor_list)

                # print(missing_person_bev_coor_arr.shape, mv_person_bev_coor_arr.shape)
                bev_dist_mat = cdist(missing_person_bev_coor_arr, mv_person_bev_coor_arr)

                for mp_idx in range(len(self.missing_objs)):
                    mp_idx_bev_dist_mat = bev_dist_mat[mp_idx]
                    match_idx = np.argmin(mp_idx_bev_dist_mat)
                    if mp_idx_bev_dist_mat[match_idx] < self.bev_dist_thresh + 0.5:
                        self.missing_objs[mp_idx].temp_bev_coor_during_missing.append(mv_person_bev_coor_arr[match_idx])


                # print(len(self.missing_objs), len(mv_persons))
                # print(bev_dist_mat)


    def find_the_temp_bev_for_missing_others(self, mv_others):

        if len(self.missing_others) and len(mv_others): # Only work when there are more than one missing person

            if len(mv_others):
                for idx_class in range(1,6,1):
                    mv_obj_indices = []
                    missing_obj_indices = []
                    for idx_mv in range(len(mv_others)):
                        if mv_others[idx_mv].obj_cls == idx_class:
                            mv_obj_indices.append(idx_mv)
                    for idx_missing_obj in range(len(self.missing_others)):
                        if self.missing_others[idx_missing_obj].obj_cls == idx_class:
                            missing_obj_indices.append(idx_missing_obj)

                    if len(mv_obj_indices) and len(missing_obj_indices):
                        mv_obj_bev_coor_list = []
                        for mv_idx in range(len(mv_obj_indices)):
                            mv_obj_bev_coor_list.append(mv_others[mv_obj_indices[mv_idx]].bev_coor)
                        mv_obj_bev_coor_arr = np.array(mv_obj_bev_coor_list)

                        missing_obj_bev_coor_list = []
                        for m_idx in range(len(missing_obj_indices)):
                            if len(self.missing_others[missing_obj_indices[m_idx]].temp_bev_coor_during_missing):
                                missing_obj_bev_coor_list.append(self.missing_others[missing_obj_indices[m_idx]].temp_bev_coor_during_missing[-1])
                            else:
                                missing_obj_bev_coor_list.append(self.missing_others[missing_obj_indices[m_idx]].last_bev_coor)
                                self.missing_others[missing_obj_indices[m_idx]].temp_bev_coor_during_missing.append(self.missing_others[missing_obj_indices[m_idx]].last_bev_coor)
                        missing_obj_bev_coor_arr = np.array(missing_obj_bev_coor_list)

                        bev_dist_mat = cdist(missing_obj_bev_coor_arr, mv_obj_bev_coor_arr)
                        # if self.cur_frameIdx > 4385:
                        #     print(self.cur_frameIdx)
                        #     print(bev_dist_mat)

                        for m_idx in range(len(missing_obj_indices)):
                            m_idx_bev_dist_mat = bev_dist_mat[m_idx]
                            match_idx = np.argmin(m_idx_bev_dist_mat)

                            # if self.cur_frameIdx > 4385:
                            #     print(self.cur_frameIdx, missing_obj_indices, m_idx)
                            #     print(mv_obj_bev_coor_arr, mv_obj_indices, m_idx)

                            if m_idx_bev_dist_mat[match_idx] < self.bev_dist_thresh + 0.5:
                                self.missing_others[missing_obj_indices[m_idx]].temp_bev_coor_during_missing.append(mv_obj_bev_coor_arr[match_idx])
                        
        return


    def update(self, dets, poses, feats, visses, frameIdx):

        self.cur_frameIdx = frameIdx
        # Remove all dets without BEV information
        bev_avai_det_indices = dets[:, 11] == 1
        dets =  dets[bev_avai_det_indices]
        poses =  poses[bev_avai_det_indices]
        feats =  feats[bev_avai_det_indices]
        visses =  visses[bev_avai_det_indices]

        # Remove all forklift dets 
        non_forklift_det_indices = dets[:, 1] != 1
        dets =  dets[non_forklift_det_indices]
        poses =  poses[non_forklift_det_indices]
        feats =  feats[non_forklift_det_indices]
        visses =  visses[non_forklift_det_indices]

        if self.cur_frameIdx == 0:
            mv_persons = self.group_person(dets, poses, feats, visses)
            for mv_person in mv_persons:
                self.tracked_objs.append(mv_person)
                self.tracked_objs[-1].global_id = self.ID_generator.get_cur_id()

            mv_others = self.group_others(dets)
            for mv_other in mv_others:
                self.tracked_others.append(mv_other)
                self.tracked_others[-1].global_id = self.ID_generator.get_cur_id()
            return 0
        else:
            person_det_indices = dets[:, 1] == 0
            person_dets = dets[person_det_indices]
            person_poses = poses[person_det_indices]
            person_feats = feats[person_det_indices]
            person_visses = visses[person_det_indices]

            self.make_too_close_tracked_missing_together()
            r_person_dets, r_person_poses, r_person_feats, r_person_visses = self.match_new_mv_persons_to_tracked_person(person_dets, person_poses, person_feats, person_visses)
            r_person_dets, r_person_poses, r_person_feats, r_person_visses = self.match_remain_person_to_tracked_person(r_person_dets, r_person_poses, r_person_feats, r_person_visses)
            r_person_dets, r_person_poses, r_person_feats, r_person_visses = self.refind_missing_person_v3(r_person_dets, r_person_poses, r_person_feats, r_person_visses)
            if self.create_new_objs: # Only work when there is no missing person
                self.create_new_persons(r_person_dets, r_person_poses, r_person_feats, r_person_visses)

            # Only work when there is at least 1 missing person and remain person dets
            self.find_the_temp_bev_for_missing_person(r_person_dets, r_person_poses, r_person_feats, r_person_visses)

            # r_person_dets, r_person_poses, r_person_feats, r_person_visses = self.refind_missing_person_v2(r_person_dets, r_person_poses, r_person_feats, r_person_visses)
            # return (len(r_person_dets))

            # For other things, exclude transporter
            other_det_indices = dets[:, 1] != 0
            other_dets = dets[other_det_indices]
            r_other_dets = self.match_new_mv_other_to_tracked_others(other_dets)
            r_other_dets = self.match_remain_other_to_tracked_others(r_other_dets)
            r_mv_others = self.refind_missing_others(r_other_dets)
            # Only work when there is at least 1 missing person and remain person dets
            self.find_the_temp_bev_for_missing_others(r_mv_others)


            return r_other_dets