import os
import sys
import glob
import json
import math
import sys

import numpy as np

from tqdm import tqdm
from loguru import logger
from easydict import EasyDict
import cv2

__ALL__ = [
	"Camera",
	"MapWorld",
	"Instance",
]


class MapWorld:

	# MARK: Magic Functions

	def __init__(self, map_cfg):
		self.map_cfg          = map_cfg
		self.map_name         = self.map_cfg['name']
		self.map_id           = self.map_cfg['id']
		self.map_type         = self.map_cfg['type'] if 'type' in self.map_cfg else None
		self.map_size         = self.map_cfg['size'] if 'size' in self.map_cfg else None
		self.calibration_path = self.map_cfg['calibration_path'] if 'calibration_path' in self.map_cfg else None
		self.groundtruth_path = self.map_cfg['groundtruth_path'] if 'groundtruth_path' in self.map_cfg else None
		self.map_image_path   = self.map_cfg['map_image'] if 'map_image' in self.map_cfg else None

		self.cameras    = {}
		self.instances  = {}

		# init cameras
		if self.calibration_path is not None:
			self.init_cameras(map_cfg, self.calibration_path)
		if self.groundtruth_path is not None:
			self.init_instances(self.groundtruth_path)
		logger.info(f"Initiate {self.map_name}.")

	# MARK: Configure

	def init_cameras(self, map_cfg, json_file):
		"""Read a JSON file containing camera calibration data and return a list of Camera objects.

		Args:
			json_file:
				{
				  "calibrationType": "cartesian",
				  "sensors": [
				    {
				      "type": "camera",
				      "id": "<sensor_id>",
				      "coordinates": {"x": float, "y": float},
				      "scaleFactor": float,
				      "translationToGlobalCoordinates": {"x": float, "y": float},
				      "attributes": [
				        {"name": "fps", "value": float},
				        {"name": "direction", "value": float},
				        {"name": "direction3d", "value": "float,float,float"},
				        {"name": "frameWidth", "value": int},
				        {"name": "frameHeight", "value": int}
				      ],
				      "intrinsicMatrix": [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]],
				      "extrinsicMatrix": [[3×4 matrix]],
				      "cameraMatrix":    [[3×4 matrix]],
				      "homography":      [[3×3 matrix]]
				    }
				  ]
				}

		Returns:
			calinration: list[Camera]
		"""
		# read json file
		with open(json_file, 'r') as f:
			data = json.load(f)

		# create and add camera objects to the calibration list
		self.cameras = {}
		for sensor_data in data["sensors"]:
			sensor_data["map_config"]  = map_cfg
			camera = Camera(sensor_data)
			self.cameras[camera.id]    = camera
			self.translation_to_global = camera.translation_to_global
			self.scale_factor          = camera.scale_factor
		# self.camera.describe()

	def init_instances(self, json_file):
		"""
			{
			  "<frame_id>": [
			    {
			      "object_type": "<class_name>",
			      "object_id": <int>,
			      "3d_location": [x, y, z],
			      "3d_bounding_box_scale": [w, l, h],
			      "3d_bounding_box_rotation": [pitch, roll, yaw],
			      "2d_bounding_box_visible": {
			        "<camera_id>": [xmin, ymin, xmax, ymax]
			      }
			    }
			  ]
			}
		Args:
			json_file:

		Returns:

		"""
		# check if the json file is exist
		if not os.path.exists(json_file):
			logger.error(f"Ground truth file {json_file} not found.")
			return

		# read json file
		with open(json_file, 'r') as f:
			data = json.load(f)

		# create and add camera objects to the calibration list
		self.instances = {}
		for frame_id, instances in tqdm(data.items(), desc=f"Loading instances {self.map_id}"):
			for instance_data in instances:
				if instance_data["object id"] not in self.instances:
					instance = Instance(instance_data)
					self.instances[instance_data["object id"]] = instance
				self.instances[instance_data["object id"]].update_trajectory(frame_id, instance_data)

		# sort instances by object id
		self.instances = dict(sorted(self.instances.items()))

	# MARK: Draw

	def draw_information_on_map(self, map_img, frame_id, color):
		"""Show the map information on the map.

		Args:
			map_img: Image to draw on
			frame_id: Frame ID to draw the instance on
			color: Arrow and text color (BGR), or chart color

		Returns:
			map_img: Image with the camera drawn on it
		"""
		# Check if frame_id is string type
		if not isinstance(frame_id, str):
			frame_id = str(frame_id)

		# NOTE: draw frame_id, panel, and object class color chart on the map
		# init values
		font              = cv2.FONT_HERSHEY_SIMPLEX
		font_scale        = 3
		thickness         = 2
		frame_id_label_tl = (5, 5)

		# Get the text size
		text_size, _ = cv2.getTextSize(frame_id, font, font_scale, thickness)
		text_width, text_height = text_size

		# Calculate the background rectangle coordinates
		x, y              = frame_id_label_tl
		top_left          = (x, y)
		bottom_right      = (x + text_width + 10, y + text_height + 10)
		frame_id_label_bl = (x, y + text_height)

		# Draw the background rectangle
		cv2.rectangle(map_img, top_left, bottom_right, (214, 224, 166), -1)

		# Draw the text with a border
		cv2.putText(map_img, frame_id, frame_id_label_bl, font, font_scale, (0 , 0 , 0), thickness + 2, cv2.LINE_AA )
		cv2.putText(map_img, frame_id, frame_id_label_bl, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA )

		# NOTE: draw object class color chart
		if isinstance(color, dict):
			# Create a color chart
			for i, (object_type, object_color) in enumerate(color.items()):
				top_left = (frame_id_label_bl[0], frame_id_label_bl[1] + 50 + i * 30)
				bottom_right = (frame_id_label_bl[0] + 100 , frame_id_label_bl[1] + 80 + i * 30)
				bottom_left  = (frame_id_label_bl[0], frame_id_label_bl[1] + 70 + i * 30)
				cv2.rectangle(map_img, top_left, bottom_right, object_color, -1)
				cv2.putText(map_img, object_type, bottom_left, font, 0.5, color=(0, 0, 0), thickness=1)

		return map_img

	def draw_cameras_on_map(self, map_img, length=30, color=(0, 0, 255)):
		"""Draw the map world with cameras.

		Args:
			map_img: Image to draw on
			length: Arrow length in pixels
			color: Arrow and text color (BGR)

		Returns:
			map_img: Image with the camera drawn on it
		"""
		for camera in self.cameras:
			map_img = self.cameras[camera].draw_on_map(map_img, length=length, color=color)
		return map_img

	def draw_instances_on_map(self, map_img, frame_id, length=20, color=(255, 255, 255)):
		"""Draw the map world with instances.

		Args:
			map_img: Image to draw on
			color: Arrow and text color (BGR)

		Returns:
			map_img: Image with the camera drawn on it
		"""
		for instance_key in self.instances:
			instance = self.instances[instance_key]
			map_img = instance.draw_on_map(map_img, frame_id, self.translation_to_global, self.scale_factor, length=length, color=color)
		return map_img

	def draw_instances_2D_on_camera(self, cam_img, camera_id, frame_id, color=(255, 255, 255)):
		"""Draw the map world with instances according to only 1 camera.

		Args:
			cam_img: Image to draw on
			color: Arrow and text color (BGR)

		Returns:
			cam_img: Image with the camera drawn on it
		"""
		for instance_key in self.instances:
			instance = self.instances[instance_key]
			cam_img  = instance.draw_2d_bboxes_on_camera_id_show(cam_img, camera_id, frame_id, color=color)
		return cam_img

	def draw_instances_on_camera(self, cam_img, camera_id, frame_id, color=(255, 255, 255)):
		"""Draw the map world with instances according to only 1 camera.

		Args:
			cam_img: Image to draw on
			color: Arrow and text color (BGR)

		Returns:
			cam_img: Image with the camera drawn on it
		"""
		for instance_key in self.instances:
			instance = self.instances[instance_key]
			cam_img  = instance.draw_2d_bboxes_on_camera_no_id_show(cam_img, self.cameras[camera_id], frame_id, color=(255, 255, 255))
			cam_img  = instance.draw_3d_bboxes_on_camera(cam_img, self.cameras[camera_id], frame_id, color=color)
		return cam_img

	def draw_instances_on_camera_and_map(self, cam_img, map_img, camera_id, frame_id, length=20, color=(255, 255, 255)):
		"""Draw the map world with instances.

		Args:
			cam_img: Image to draw on
			color: Arrow and text color (BGR)

		Returns:
			cam_img: Image with the camera drawn on it
		"""
		for instance_key in self.instances:
			instance = self.instances[instance_key]
			map_img  = instance.draw_on_map_in_one_camera(map_img, camera_id, frame_id, self.translation_to_global, self.scale_factor, length=length, color=color)
			cam_img  = instance.draw_2d_bboxes_on_camera_no_id_show(cam_img, self.cameras[camera_id], frame_id, color=(255, 255, 255))
			cam_img  = instance.draw_3d_bboxes_on_camera(cam_img, self.cameras[camera_id], frame_id, color=color)
		return cam_img, map_img


class Camera:

	# MARK: Magic Functions

	def __init__(self, sensor_data):
		"""
		Initialize a Camera object with sensor data.

		Args:
		    sensor_data (dict): A dictionary containing camera sensor data, including:
		        - id (str): Camera ID.
		        - coordinates (dict): Camera coordinates with keys 'x' and 'y'.
		        - scaleFactor (float): Scale factor for the camera.
		        - translationToGlobalCoordinates (dict): Translation to global coordinates with keys 'x' and 'y'.
		        - intrinsicMatrix (list): 3x3 intrinsic matrix.
		        - extrinsicMatrix (list): 3x4 extrinsic matrix.
		        - cameraMatrix (list): 3x4 camera matrix.
		        - homography (list): 3x3 homography matrix.
		        - attributes (list): List of attribute dictionaries with 'name' and 'value' keys.
		"""
		self.id                    = sensor_data.get('id')
		self.camera_file_name      = sensor_data.get('id')
		self.coordinates           = sensor_data.get('coordinates')
		self.scale_factor          = sensor_data.get('scaleFactor')
		self.translation_to_global = sensor_data.get('translationToGlobalCoordinates')
		self.intrinsic_matrix      = sensor_data.get('intrinsicMatrix')
		self.extrinsic_matrix      = sensor_data.get('extrinsicMatrix')
		self.camera_matrix         = sensor_data.get('cameraMatrix')
		self.homography            = sensor_data.get('homography')

		# camera matrix P (3x4)
		camera_matrix = np.array(self.camera_matrix)

		# Extract rotation matrix (first 3 columns)
		self.rotation_matrix = camera_matrix[:, :3].tolist()

		# Extract translation vector (last column)
		self.translation_vector = camera_matrix[:, 3].reshape((3, 1))

		# Change camera_id into Camera_XXXX
		self.id = self.adjust_camera_id(self.id)

		# Extract attributes into a dictionary
		self.attributes   = {attr['name']: attr['value'] for attr in sensor_data.get('attributes', [])}
		self.fps          = float(self.attributes.get('fps'))
		self.direction    = float(self.attributes.get('direction'))
		self.direction3d  = [float(value) for value in self.attributes.get('direction3d').split(',')]
		self.frame_width  = int(self.attributes.get('frameWidth'))
		self.frame_height = int(self.attributes.get('frameHeight'))

	def __repr__(self):
		return (f"<Camera id={self.id}, fps={self.fps}, resolution=({self.frame_width}x{self.frame_height}), "
		        f"direction={self.direction}>")

	def describe(self):
		print(f"Camera ID: {self.id}")
		print(f"Coordinates: {self.coordinates}")
		print(f"Scale Factor: {self.scale_factor}")
		print(f"Translation to Global: {self.translation_to_global}")
		print(f"FPS: {self.fps}")
		print(f"Frame Size: {self.frame_width} x {self.frame_height}")
		print(f"Direction: {self.direction}")
		print(f"Direction3D: {self.direction3d}")
		print(f"Intrinsic Matrix: {self.intrinsic_matrix}")
		print(f"Extrinsic Matrix: {self.extrinsic_matrix}")
		print(f"Camera Matrix: {self.camera_matrix}")
		print(f"Rotation Matrix: {self.rotation_matrix}")
		print(f"Translation Vector: {self.translation_vector}")
		print(f"Homography: {self.homography}")
		print(f"Attributes: {self.attributes}")
		print("-" * 50)

	# MARK: Configure

	@staticmethod
	def adjust_camera_id(camera_id):
		"""Check if the camera ID is valid."""
		# Change camera_id into Camera_XXXX
		camera_name = camera_id.replace('_','').replace('Camera','')
		if camera_name == "":
			camera_id = "Camera_0000"
		else:
			camera_id = str(f"Camera_{int(camera_name):04d}")
		return camera_id

	# MARK: Functions

	# MARK: Draw

	def draw_on_map(self, map_img, length=30, color=(0, 0, 255)):
		"""Draw the camera on the map using coordinates, translation_to_global, and scale_factor.

		Args:
			map_img: Image to draw on
			length: Arrow length in pixels
			color: Arrow and text color (BGR)

		Returns:
			map_img: Image with the camera drawn on it
		"""
		if not self.coordinates or self.translation_to_global is None or self.scale_factor is None:
			print(f"Camera {self.id}: Missing coordinates, translation, or scale.")
			return map_img

		# Compute world position in pixel coordinates
		x_px = int((self.coordinates["x"] + self.translation_to_global["x"]) * self.scale_factor)
		y_px = int((self.coordinates["y"] + self.translation_to_global["y"]) * self.scale_factor)

		# Compute direction arrow based on rotation angle
		if self.direction is None:
			print(f"Camera {self.id}: Missing direction angle.")
			return map_img

		# FIXME: get direction
		angle_rad = math.radians(self.direction) if abs(self.direction) > 2 * math.pi else self.direction
		dx = int(math.sin(angle_rad) * length)
		dy = int(math.cos(angle_rad) * length)

		# from the (0,0) is the top-left corner to bottom-left corner of the image
		img_h, img_w, _ = map_img.shape
		y_px = img_h - y_px  # Invert y coordinate
		pt1  = (x_px       , y_px)
		pt2  = (x_px + dx  , y_px - dy)

		# Draw the arrow and label
		camera_name = str(int(self.id.replace('_','').replace('Camera','')))
		cv2.arrowedLine(map_img, pt1, pt2, color, thickness=3, tipLength=0.5)
		# cv2.putText(map_img, f"{camera_name}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		map_img = put_text_with_border(map_img, f"{camera_name}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=color, border_color=(0,0,0), thickness=1)
		return map_img


class Instance:

	# MARK: Magic Functions
	def __init__(self, instance_data):
		"""
	    Initialize an Instance object with instance data.

	    Args:
	        instance_data (dict): A dictionary containing instance data, including:
	            - "object type" (str): The type of the object
	                ("Person", "Forklift", "NovaCarter", "Transporter", "FourierGR1T2", "AgilityDigit").
	            - "object id" (int): The unique identifier for the object.
	    """
		self.object_type     = instance_data.get("object type")
		self.object_id       = instance_data.get("object id")
		self.frames          = {}

		self.frame_id_min    = sys.maxsize
		self.frame_id_max    = 0

	def __repr__(self):
		return f"<Instance type={self.object_type}, id={self.object_id}>"

	# MARK: Configure

	# MARK: methods

	def update_bbox(self, instance_data):
		"""
		Update the bounding box for a specific frame.

		Args:
			frame_id: Frame ID to update
			instance_data: Instance data containing bounding box information
				instance_data = {
						"camera_id"  : camera_name,
						"frame_id"   : int(parts[0]),
						"track_id"   : int(parts[1]),  # object id
						"x_tl"       : float(parts[2]),
						"y_tl"       : float(parts[3]),
						"x_br"       : float(parts[2]) + float(parts[4]),
						"y_br"       : float(parts[3]) + float(parts[5]),
						"w"          : float(parts[4]),
						"h"          : float(parts[5]),
						"not_ignored": int(parts[6]),
						"class_id"   : int(parts[7]),
						"visibility" : float(parts[8]),
					}
		Returns:
			None
		"""
		# check if the frame_id is string type
		if not isinstance(instance_data["frame_id"], str):
			frame_id = str(instance_data["frame_id"])

		# check if the frame_id is in the instance
		if frame_id not in self.frames:
			self.frames[frame_id] = EasyDict({
				"location_3d"    : [],
				"scale_3d"       : [],
				"rotation_3d"    : [],
				"bbox_visible_2d": {}
			})

		# add 2d bounding box
		self.frames[frame_id]["bbox_visible_2d"].update({
			instance_data["camera_id"] : [instance_data["x_tl"], instance_data["y_tl"], instance_data["x_br"], instance_data["y_br"]]
		})

	def update_trajectory(self, frame_id, instance_data):
		"""
			{
			  "<frame_id>": [
			    {
			      "object_type": "<class_name>",
			      "object_id": <int>,
			      "3d_location": [x, y, z],
			      "3d_bounding_box_scale": [w, l, h],
			      "3d_bounding_box_rotation": [pitch, roll, yaw],
			      "2d_bounding_box_visible": {
			        "<camera_id>": [xmin, ymin, xmax, ymax]
			      }
			    }
			  ]
			}
		Returns:

		"""
		# Adjust camera id into Camera_XXXX
		bbox_visible_2d = {}
		for camera_id, bbox in instance_data.get("2d bounding box visible").items():
			camera_id = Camera.adjust_camera_id(camera_id)
			bbox_visible_2d[camera_id] = bbox

		# update frame data
		self.frames[frame_id] = EasyDict({
			"location_3d"    : instance_data.get("3d location"),
			"scale_3d"       : instance_data.get("3d bounding box scale"),
			"rotation_3d"    : instance_data.get("3d bounding box rotation"),
			"bbox_visible_2d": bbox_visible_2d
		})

		# update frame id min and max
		self.frame_id_min = min(self.frame_id_min, int(frame_id))
		self.frame_id_max = max(self.frame_id_max, int(frame_id))

	def sort_frames(self):
		self.frames = dict(sorted(self.frames.items()))

	@staticmethod
	def get_3d_bounding_box_on_2d_image_coordinate(location_3d, scale_3d, rotation_3d, intrinsic_matrix, extrinsic_matrix):

		def convert_3d_to_2d(point3d, intrinsic_matrix, extrinsic_matrix):
			point3d1      = np.array([point3d[0], point3d[1], point3d[2], 1]).reshape(4, 1)
			camera_point  = extrinsic_matrix @ point3d1     # shape (3, 1)
			pixel_point   = intrinsic_matrix @ camera_point # shape (3, 1)
			pixel_point  /= pixel_point[2]                  # normalize
			return np.array([pixel_point[:2]])

		# get pitch, roll, and yaw
		pitch, roll, yaw = rotation_3d

		# DEBUG:
		# yaw = 0
		pitch = 0
		roll  = 0

		# size of the bouding box project to each dimention (x, y, z)
		x = (scale_3d[0] / 2)
		y = (scale_3d[1] / 2)
		z = (scale_3d[2] / 2)

		# TEST:
		# yaw = yaw % math.pi
		# l = ((x * math.sin(yaw)) - (y * math.cos(yaw))) / ((math.sin(yaw) * math.sin(yaw)) - (math.cos(yaw) * math.cos(yaw)))
		# w = (x - (l * math.sin(yaw))) / math.cos(yaw)
		# x = w
		# y = l

		corners = np.array([
			[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],  # Bottom face
			[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]       # Top face
		])

		# corners_arrow_3d = np.array([[0, 0, -z], [-x, 0 ,-z]])adasdasd

		# Create rotation matrices for pitch, roll, and yaw
		# pitch, roll, yaw = rotation_3d
		Rx = np.array([
			[1,              0,               0],
			[0, math.cos(roll),  -math.sin(roll)],
			[0, math.sin(roll),  math.cos(roll)]
		])
		Ry = np.array([
			[math.cos(pitch),  0, math.sin(pitch)],
			[0,                1,               0],
			[-math.sin(pitch), 0, math.cos(pitch)]
		])
		# FIXME: sincos lai thay doi vi tri o day
		# yaw = yaw % math.pi
		# yaw = -yaw
		Rz = np.array([
			[math.cos(yaw)  ,   -math.sin(yaw), 0],
			[math.sin(yaw) ,   math.cos(yaw), 0],
			[              0,               0, 1]
		])

		# Combine rotations into a single matrix
		R = Rz @ Ry @ Rx

		# Rotate the corners
		rotated_corners          = (R @ corners.T).T
		# rotated_corners_arrow_3d = (R @ corners_arrow_3d.T).T


		# Translate to the final position
		box_3d   = rotated_corners + np.array(location_3d)
		# box_3d = corners + np.array(location_3d)
		# arrow_3d = rotated_corners_arrow_3d + np.array(location_3d)

		proj_box3d   = np.zeros((8,2,1))
		# proj_arrow3d = np.zeros((2,2,1))

		for idx in range(0, 8):
			proj_box3d[idx] = convert_3d_to_2d(box_3d[idx], intrinsic_matrix, extrinsic_matrix)[0]

		# for idx in range(0, 2):
		# 	proj_arrow3d[idx] = convert_3d_to_2d(arrow_3d[idx], intrinsic_matrix, extrinsic_matrix)[0]

		# return proj_box3d, proj_arrow3d
		return proj_box3d, box_3d

	# MARK: draw

	def draw_2d_bboxes_on_camera_id_show(self, cam_img, camera_id, frame_id, color=(255, 255, 255)):
		"""
		Draws the 2D bounding box on the camera image for a specific frame.

		Args:
			camera_id: The camera ID to visualize the instance.
			cam_img: The image to draw on.
			frame_id: The frame ID to visualize the instance.
			color: Color for drawing the instance (BGR tuple or color chart dict).

		Returns:
			cam_img: The image with the instance drawn on it.
		"""
		# check camera id
		if camera_id is None:
			# print(f"Camera ID is None.")
			return cam_img

		# check color for the instance
		if isinstance(color, dict):
			if self.object_type in color:
				color = color[self.object_type]
			else:
				color = (255, 255, 255)

		# check if the frame_id is string type
		if not isinstance(frame_id, str):
			frame_id = str(frame_id)

		# check if the frame_id is in the instance
		if frame_id not in self.frames:
			# print(f"Instance {self.object_id} not found at Camera ID {camera_id} at frame {frame_id}.")
			return cam_img

		# check if the camera_id is in the instance
		if camera_id not in self.frames[frame_id]["bbox_visible_2d"]:
			# print(f"Instance {self.object_id} not found at Camera ID {camera_id} at frame {frame_id}.")
			return cam_img

		frame_data = self.frames[frame_id]
		bbox_2d    = np.array(frame_data["bbox_visible_2d"][camera_id], dtype=int)

		# draw bounding box
		x1, y1, x2, y2 = bbox_2d
		cv2.rectangle(cam_img, (x1, y1), (x2, y2), color, thickness=4)

		# ngochdm
		# cam_img = put_text_with_border(cam_img, f"{self.object_id}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color=color, border_color=(0,0,0), thickness=2)
		cam_img = put_text_with_border(cam_img, f"{self.object_id}", (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color=color, border_color=(0,0,0), thickness=2)
		return cam_img

	def draw_2d_bboxes_on_camera_no_id_show(self, cam_img, camera, frame_id, color=(255, 255, 255)):
		"""
		Draws the 2D bounding box on the camera image for a specific frame.

		Args:
			camera: The camera object to visualize the instance.
			cam_img: The image to draw on.
			frame_id: The frame ID to visualize the instance.
			color: Color for drawing the instance (BGR tuple or color chart dict).

		Returns:
			cam_img: The image with the instance drawn on it.
		"""
		# check camera object
		if not isinstance(camera, Camera):
			print(f"Camera {camera.id} is not a valid Camera object.")
			return cam_img
		camera_id = camera.id

		# check color for the instance
		if isinstance(color, dict):
			if self.object_type in color:
				color = color[self.object_type]
			else:
				color = (255, 255, 255)

		# check if the frame_id is string type
		if not isinstance(frame_id, str):
			frame_id = str(frame_id)

		# check if the frame_id is in the instance
		if frame_id not in self.frames:
			# print(f"Instance {self.object_id} not found at Camera ID {camera_id} at frame {frame_id}.")
			return cam_img

		# check if the camera_id is in the instance
		if camera_id not in self.frames[frame_id]["bbox_visible_2d"]:
			# print(f"Instance {self.object_id} not found at Camera ID {camera_id} at frame {frame_id}.")
			return cam_img

		frame_data = self.frames[frame_id]
		bbox_2d    = np.array(frame_data["bbox_visible_2d"][camera_id], dtype=int)

		# draw bounding box
		x1, y1, x2, y2 = bbox_2d
		cv2.rectangle(cam_img, (x1, y1), (x2, y2), color, thickness=4)
		return cam_img

	def draw_3d_bboxes_on_camera(self, cam_img, camera, frame_id, color=(255, 255, 255)):
		"""
		Draws the in stance on the camera image for a specific frame.

		Args:
			camera: The camera object to visualize the instance.
			cam_img: The image to draw on.
			frame_id: The frame ID to visualize the instance.
			color: Color for drawing the instance (BGR tuple or color chart dict).

		Returns:
			cam_img: The image with the instance drawn on it.
		"""
		# get image size
		img_h, img_w, _ = cam_img.shape

		# check camera object
		if not isinstance(camera, Camera):
			print(f"Camera {camera.id} is not a valid Camera object.")
			return cam_img

		# check color for the instance
		if isinstance(color, dict):
			if self.object_type in color:
				color = color[self.object_type]
			else:
				color = (255, 255, 255)

		if not isinstance(frame_id, str):
			frame_id = str(frame_id)

		if frame_id not in self.frames:
			# print(f"Instance {self.object_id} not found at any CAMERA at frame {frame_id}.")
			return cam_img

		# check instance in Camera or not
		if camera.id not in self.frames[frame_id]["bbox_visible_2d"]:
			# print(f"Instance {self.object_id} not found Camera ID {camera.id} at frame {frame_id}.")
			return cam_img

		frame_data = self.frames[frame_id]

		if (frame_data["location_3d"] is None or frame_data["scale_3d"] is None or frame_data["rotation_3d"] is None
				or len(frame_data["location_3d"]) != 3 or len(frame_data["scale_3d"]) != 3 or len(frame_data["rotation_3d"]) != 3):
			# print(f"Instance {self.object_id} not found at Camera ID {camera.id} at frame {frame_id}.")
			return cam_img


		# proj_box3d, proj_arrow3d = self.get_3d_bounding_box_on_2d_image_coordinate(frame_data["location_3d"], frame_data["scale_3d"], frame_data["rotation_3d"], camera.intrinsic_matrix, camera.extrinsic_matrix)
		proj_box3d, _ = self.get_3d_bounding_box_on_2d_image_coordinate(frame_data["location_3d"], frame_data["scale_3d"], frame_data["rotation_3d"], camera.intrinsic_matrix, camera.extrinsic_matrix)

		# DEBUG:
		# if frame_id in ["1", "450"]:
		# 	if self.object_id in [95]:
		# 		print(f"\n\n  {self.object_id=}--{frame_id=}--{frame_data['rotation_3d']=}--{frame_data['scale_3d']=}--{math.prod(frame_data['scale_3d'])}\n\n")

		# Define edges to connect
		edges = [
			(0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
			(4, 5), (5, 6), (6, 7), (7, 4),  # top square
			(0, 4), (1, 5), (2, 6), (3, 7)   # vertical lines
		]

		# draw 3d bounding box
		for start, end in edges:
			pt1 = (int(proj_box3d[start][0]), int(proj_box3d[start][1]))
			pt2 = (int(proj_box3d[end][0]),   int(proj_box3d[end][1]))
			try:
				cv2.line(cam_img, pt1, pt2, color, thickness=2)
			except cv2.error as e:
				logger.error(e)

		# draw 3d bounding box arrow
		# pt1 = (int(proj_arrow3d[0][0]), int(proj_arrow3d[0][1]))
		# pt2 = (int(proj_arrow3d[1][0]), int(proj_arrow3d[1][1]))
		# cv2.arrowedLine(cam_img, pt1, pt2, color, thickness=3, tipLength=0.5)

		# Add label
		x_px = int(proj_box3d[5][0])
		y_px = int(proj_box3d[5][1])
		cam_img = put_text_with_border(cam_img, f"{self.object_id}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color=color, border_color=(0,0,0), thickness=5)

		return cam_img

	def draw_on_map(self, map_img, frame_id, translation_to_global, scale_factor, length=20, color=(255, 255, 255)):
		"""
		Draws the instance on the map image for a specific frame.

		Args:
			map_img: The image to draw on.
			frame_id: The frame ID to visualize the instance.
			translation_to_global: Dictionary with 'x' and 'y' translation to global coordinates.
			scale_factor: Scale factor to convert world coordinates to pixel coordinates.
			length: Length of the direction arrow (default: 20).
			color: Color for drawing the instance (BGR tuple or color chart dict).

		Returns:
			map_img: The image with the instance drawn on it.
		"""
		# Check if frame_id is string type
		if not isinstance(frame_id, str):
			frame_id = str(frame_id)

		# check if the frame_id is in the instance
		if frame_id not in self.frames:
			# print(f"Instance {self.object_id}: Frame ID {frame_id} not found.")
			return map_img

		frame_data = self.frames[frame_id]
		# Compute world position in pixel coordinates
		x_px = int((frame_data["location_3d"][0] + translation_to_global["x"]) * scale_factor)
		y_px = int((frame_data["location_3d"][1] + translation_to_global["y"]) * scale_factor)

		# FIXME: get direction
		angle_rad = math.radians(frame_data["rotation_3d"][2]) if abs(frame_data["rotation_3d"][2]) > 2 * math.pi else frame_data["rotation_3d"][2]
		dx = int(math.sin(angle_rad) * length)
		dy = int(math.cos(angle_rad) * length)

		# from the (0,0) is the top-left corner to bottom-left corner of the image
		img_h, img_w, _ = map_img.shape
		y_px = img_h - y_px  # Invert y coordinate
		pt1  = (x_px       , y_px)
		pt2  = (x_px - dx  , y_px + dy)

		# check color for the instance
		if isinstance(color, dict):
			if self.object_type in color:
				color = color[self.object_type]
			else:
				color = (255, 255, 255)

		# Draw the instance and label
		cv2.arrowedLine(map_img, pt1, pt2, color, thickness=3, tipLength=0.5)
		cv2.circle(map_img, (x_px, y_px), 5, color, -1)
		# cv2.putText(map_img, f"{self.object_id}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		map_img = put_text_with_border(map_img, f"{self.object_id}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=color, border_color=(0,0,0), thickness=1)
		return map_img

	def draw_on_map_in_one_camera(self, map_img, camera_id, frame_id, translation_to_global, scale_factor, length=20, color=(255, 255, 255)):
		"""
		Draws the instance on the map image for a specific frame in SPECIFIC CAMERA.

		Args:
			map_img: The image to draw on.
			camera_id: The camera ID to visualize the instance.
			frame_id: The frame ID to visualize the instance.
			translation_to_global: Dictionary with 'x' and 'y' translation to global coordinates.
			scale_factor: Scale factor to convert world coordinates to pixel coordinates.
			length: Length of the direction arrow (default: 20).
			color: Color for drawing the instance (BGR tuple or color chart dict).

		Returns:
			map_img: The image with the instance drawn on it.
		"""
		# Check if frame_id is string type
		if not isinstance(frame_id, str):
			frame_id = str(frame_id)

		# check if the frame_id is in the instance
		if frame_id not in self.frames:
			# print(f"Instance {self.object_id}: Frame ID {frame_id} not found.")
			return map_img

		# NOTE: check this object is can be found in this camera
		if camera_id not in self.frames[frame_id]["bbox_visible_2d"]:
			# print(f"Instance {self.object_id} not found Camera ID {camera.id} at frame {frame_id}.")
			return map_img

		frame_data = self.frames[frame_id]
		# Compute world position in pixel coordinates
		x_px = int((frame_data["location_3d"][0] + translation_to_global["x"]) * scale_factor)
		y_px = int((frame_data["location_3d"][1] + translation_to_global["y"]) * scale_factor)

		# FIXME: get direction
		angle_rad = math.radians(frame_data["rotation_3d"][2]) if abs(frame_data["rotation_3d"][2]) > 2 * math.pi else frame_data["rotation_3d"][2]
		dx = int(math.sin(angle_rad) * length)
		dy = int(math.cos(angle_rad) * length)

		# from the (0,0) is the top-left corner to bottom-left corner of the image
		img_h, img_w, _ = map_img.shape
		y_px = img_h - y_px  # Invert y coordinate
		pt1  = (x_px       , y_px)
		pt2  = (x_px - dx  , y_px + dy)

		# check color for the instance
		if isinstance(color, dict):
			if self.object_type in color:
				color = color[self.object_type]
			else:
				color = (255, 255, 255)

		# Draw the instance and label
		cv2.arrowedLine(map_img, pt1, pt2, color, thickness=3, tipLength=0.5)
		cv2.circle(map_img, (x_px, y_px), 5, color, -1)
		# cv2.putText(map_img, f"{self.object_id}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		map_img = put_text_with_border(map_img, f"{self.object_id}", (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color=color, border_color=(0,0,0), thickness=1)
		return map_img


# MARK: Additional Functions

def put_text_with_border(image, text, org, font, font_scale, text_color, border_color, thickness):
	"""
	Adds text with a border to an image.

	Args:
		image: The image to draw on.
		text: The text string to write.
		org: The bottom-left corner coordinates of the text string.
		font: Font type, e.g., cv2.FONT_HERSHEY_SIMPLEX.
		font_scale: Font scale factor.
		text_color: Text color in BGR format (e.g., (255, 255, 255) for white).
		border_color: Border color in BGR format (e.g., (0, 0, 0) for black).
		thickness: Thickness of the text and border.
	"""
	# Draw border by drawing the text first with the border color and a larger thickness
	cv2.putText(image, text, org, font, font_scale, border_color, thickness + 2, cv2.LINE_AA)
	# Draw the actual text on top of the border
	cv2.putText(image, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)
	return image
