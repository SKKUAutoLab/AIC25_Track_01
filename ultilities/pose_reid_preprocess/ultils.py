import os
import cv2
import json
from pathlib import Path

def generate_nested_folders(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)

def mkdir_if_not(pth):
    if os.path.exists(pth) is False:
        os.mkdir(pth)


def read_video(video_folder_path):
    # Read video
    cameras = os.listdir(video_folder_path)
    camera_names = []
    camera_feeds = []
    for camera in cameras:
        video_file = video_folder_path + camera
        video_capture = cv2.VideoCapture(video_file)
        camera_names.append(camera.split('.')[0])
        camera_feeds.append(video_capture)

    return camera_names, camera_feeds


def read_gt_json(gt_json_path, as_person_key=False):

    with open(gt_json_path, 'r') as f:
        data = json.load(f)
        
        if as_person_key is False:
            return data
        person_dict = {}
        for idx in data.keys():
            frame_data = data[idx]
            for obj in frame_data:
                if obj['object type'] == 'Person':
                    if obj['object id'] in person_dict:
                        person_dict[obj['object id']].append({
                            'frame_id': obj['object id'],
                            '2d bounding box visible': obj['2d bounding box visible']
                        })
                    else:
                        person_dict[obj['object id']] = [
                            {
                                'frame_id': obj['object id'],
                                '2d bounding box visible': obj['2d bounding box visible']
                            }
                        ]

        return person_dict


def read_calibration(calib_path):

    calib_dict = {}
    new_dict_by_cam_key = {}
    with open(calib_path, 'r') as f:
        data = json.load(f)
        # print(data['sensors'][0])
        for item in data['sensors']:
            if item['type'] == 'camera':
                new_dict_by_cam_key[item['id']] = {
                    'coordinates': item['coordinates'],
                    'scaleFactor': item['scaleFactor'],
                    'translationToGlobalCoordinates': item['translationToGlobalCoordinates'],
                    'intrinsicMatrix': item['intrinsicMatrix'],
                    'extrinsicMatrix': item['extrinsicMatrix'],
                    'cameraMatrix': item['cameraMatrix'],
                    'homography': item['homography'],
                }
    return new_dict_by_cam_key


def sort_points_clockwise(points):
    """
    Sort 2D points in clockwise order around their centroid.
    """
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def polygon_area_unordered(points, pose_score_threshold):
    """
    Estimate the area of a polygon from unordered 2D points.
    """
    ordered_points = sort_points_clockwise(points)
    x = ordered_points[ordered_points[:, 2] > pose_score_threshold][:, 0]
    y = ordered_points[ordered_points[:, 2] > pose_score_threshold][:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def get_cam_id_from_cam_name(cam_name):
    id_text = cam_name.split('_')[-1]
    cam_id = 0
    if id_text != 'Camera':
        cam_id = int(id_text)
    return cam_id


def tlbr_to_xywh_safe(bbox, image_shape):
    """
    Convert (x1, y1, x2, y2) to (x, y, w, h) and validate it.

    Args:
        bbox (list or array): [x1, y1, x2, y2]
        image_shape (tuple): (height, width)

    Returns:
        list: [x, y, w, h] or raises ValueError if invalid
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = image_shape

    # Clamp bbox to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False, [0, 0, 0, 0]

    return True, [x1, y1, w, h]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def get_overlap_region(box1, box2):
    # x1_min, y1_min, x1_max, y1_max = xywh_to_xyxy(box1)
    # x2_min, y2_min, x2_max, y2_max = xywh_to_xyxy(box2)

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    if xi2 > xi1 and yi2 > yi1:
        # Return overlap in [x, y, w, h]
        return [xi1, yi1, xi2 - xi1, yi2 - yi1]
    else:
        return None  # No overlap


def point_in_box(point, box):
    px, py = point
    bx, by, bw, bh = box

    return bx <= px <= bx + bw and by <= py <= by + bh


def draw_crop_viz(crop_img, json_data):
    
    for adict in json_data:
        kps = adict['keypoints']
        is_target = adict['is_target']

        a_color = (0, 0, 255)
        if is_target == True:
            a_color = (0, 255, 0)
        
        for kp in kps:
            cv2.circle(crop_img, (int(kp[0]), int(kp[1])),  
                    radius=3, color=a_color, thickness=-1, lineType=cv2.LINE_AA)

    return crop_img

