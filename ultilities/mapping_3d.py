import argparse
import base64
import json
import math
import os
import shutil
import sys
import glob
from copy import deepcopy
import time
import threading
from functools import cmp_to_key

import cv2
import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

import open3d as o3d
from scipy.spatial.transform import Rotation as R

from ultilities import configuration

# region configuration

scene_id_table ={
	"Train" : {
		0 : "Warehouse_000",
		1 : "Warehouse_001",
		2 : "Warehouse_002",
		3 : "Warehouse_003",
		4 : "Warehouse_004",
		5 : "Warehouse_005",
		6 : "Warehouse_006",
		7 : "Warehouse_007",
		8 : "Warehouse_008",
		9 : "Warehouse_009",
		10: "Warehouse_010",
		11: "Warehouse_011",
		12: "Warehouse_012",
		13: "Warehouse_013",
		14: "Warehouse_014",
	},
	"Val" : {
		15: "Warehouse_015",
		16: "Warehouse_016",
		22: "Lab_000",
		23: "Hospital_000",
	},
	"Test" : {
		17: "Warehouse_017",
		18: "Warehouse_018",
		19: "Warehouse_019",
		20: "Warehouse_020",
	}
}


object_type_name = {
	0 : "Person", # green
	1 : "Forklift", # green
	2 : "NovaCarter", # pink
	3 : "Transporter", # yellow
	4 : "FourierGR1T2", # purple
	5 : "AgilityDigit", # blue
}


color_chart = {
	"Person"      : (77, 109, 163), # brown
	"Forklift"    : (162, 245, 214), # light yellow
	"NovaCarter"  : (245, 245, 245), # light pink
	"Transporter" : (0  , 255, 255), # yellow
	"FourierGR1T2": (164, 17 , 157), # purple
	"AgilityDigit": (235, 229, 52) , # blue
}

object_type_id = {
	"Person"       : 0, # green
	"Forklift"     : 1, # green
	"NovaCarter"   : 2, # pink
	"Transporter"  : 3, # yellow
	"FourierGR1T2" : 4, # purple
	"AgilityDigit" : 5, # blue
}

def find_scene_id(scene_name):
	"""
	Finds the scene ID based on the given scene name.
	"""
	for split in scene_id_table:
		for scene_id, name in scene_id_table[split].items():
			if name == scene_name:
				return scene_id
	logger.error(f"Scene name {scene_name} not found in any dataset split.")
	return None

def load_final_result(final_result_path, scene_id):
	def custom_final_result_sort(part_a, part_b):
		"""frame_id->object_id->class_id"""
		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		if part_a[3] != part_b[3]:
			return int(part_a[3]) - int(part_b[3])
		if part_a[2] != part_b[2]:
			return int(part_a[2]) - int(part_b[2])
		if part_a[1] != part_b[1]:
			return int(part_a[1]) - int(part_b[1])
		return 0
	# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
	final_result = []
	with open(final_result_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if line.startswith("#") or line.strip() == "":
				continue
			if "," in line:
				parts = np.array(line.split(","), dtype=np.float32)
			else:
				parts = np.array(line.split(), dtype=np.float32)
			if int(parts[0]) == scene_id:
				final_result.append(parts)

	return sorted(final_result, key=cmp_to_key(custom_final_result_sort))

class json_serialize(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

# endregion

# region prepare data

def create_lookup_table(scene_name, scene_names_lookup, folder_input, folder_output):
	# Create the output folder if it does not exist
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)

	# create lookup table with object types
	lookup_table = {}
	for type_id, type_name in object_type_name.items():
		# Create a dictionary to hold the lookup table for this object type
		lookup_table[type_name] = {
			"type_id"    : type_id,
			"color"      : color_chart[type_name],
			"shape_max": { "width": 0           , "length": 0           , "height": 0 }          , # [width, length, height]
			"shape_min": { "width": float('inf'), "length": float('inf'), "height": float('inf')}, # [width, length, height]
			"shape_avg": { "width": 0.0         , "length": 0.0         , "height": 0.0 }        , # [width, length, height]
			"shape_count": 0,  # Count of shapes for averaging
			"shapes"     : []
		}


	# Create the lookup table for the specific scene
	index_count = 0
	for scene, object_type_specs in tqdm(scene_names_lookup, desc=f"Creating lookup table for {scene_name}"):
		json_path_groundtruth = glob.glob(os.path.join(folder_input, f"*/{scene}/ground_truth.json"))[0]
		json_path_calibration = glob.glob(os.path.join(folder_input, f"*/{scene}/calibration.json"))[0]
		with open(json_path_groundtruth, 'r') as f:
			json_data_groundtruth = json.load(f)

		with open(json_path_calibration, 'r') as f:
			json_data_calibration = json.load(f)

		scale_factor =  json_data_calibration["sensors"][0]["scaleFactor"]

		for frame_id, frame_data in tqdm(json_data_groundtruth.items(), desc=f"Processing creating lookup table in {scene}"):
			for object_instance in frame_data:
				object_id       = object_instance["object id"]
				object_type     = object_instance["object type"]
				object_shape    = object_instance["3d bounding box scale"]
				object_roation  = object_instance["3d bounding box rotation"]
				object_location = object_instance["3d location"]

				# DEBUG:
				# print(object_type)

				if object_type not in object_type_specs:
					continue

				# DEBUG: reduce the number person for lookup table
				# FIXME: remove person in Warehouse_001, Warehouse_003, Warehouse_004
				# if (object_type.lower() == str("Person").lower()
				# 	and scene in ["Warehouse_001","Warehouse_003", "Warehouse_004"]):
				# 		continue

				# {width, length, height, pitch, roll ,yaw}
				lookup_table[object_type]["shapes"].append({
					"x"     : object_location[0],
					"y"     : object_location[1],
					"z"     : object_location[2],
					"width" : object_shape[0],
					"length": object_shape[1],
					"height": object_shape[2],
					"pitch" : object_roation[0],
					"roll"  : object_roation[1],
					"yaw"   : object_roation[2],
				})

				# find max min, height for agility digit and fourier gr1t2
				# [width, length, height]
				lookup_table[object_type]["shape_max"]["width"] = max(lookup_table[object_type]["shape_max"]["width"],object_shape[0])
				lookup_table[object_type]["shape_min"]["width"] = min(lookup_table[object_type]["shape_min"]["width"],object_shape[0])
				lookup_table[object_type]["shape_avg"]["width"] += object_shape[0]
				lookup_table[object_type]["shape_max"]["length"] = max(lookup_table[object_type]["shape_max"]["length"],object_shape[1])
				lookup_table[object_type]["shape_min"]["length"] = min(lookup_table[object_type]["shape_min"]["length"],object_shape[1])
				lookup_table[object_type]["shape_avg"]["length"] += object_shape[1]
				lookup_table[object_type]["shape_max"]["height"] = max(lookup_table[object_type]["shape_max"]["height"],object_shape[2])
				lookup_table[object_type]["shape_min"]["height"] = min(lookup_table[object_type]["shape_min"]["height"],object_shape[2])
				lookup_table[object_type]["shape_avg"]["height"] += object_shape[2]
				lookup_table[object_type]["shape_count"] += 1


	for type_id, type_name in object_type_name.items():
		if lookup_table[type_name]["shape_count"] > 0:
			lookup_table[type_name]["shape_avg"]["width"]  /= lookup_table[type_name]["shape_count"]
			lookup_table[type_name]["shape_avg"]["length"] /= lookup_table[type_name]["shape_count"]
			lookup_table[type_name]["shape_avg"]["height"] /= lookup_table[type_name]["shape_count"]

	# DEBUG: print max min height for agility digit and fourier gr1t2
	for type_id, type_name in object_type_name.items():
		print(f"{type_name} shape ::::: \n "
		      f"max -- {lookup_table[type_name]['shape_max']}, \n "
		      f"min -- {lookup_table[type_name]['shape_min']}, \n "
		      f"avg -- {lookup_table[type_name]['shape_avg']}, \n "
		      f"count -- {lookup_table[type_name]['shape_count']}")


	# Save the lookup table to a JSON file
	output_json_path = os.path.join(folder_output, f"{scene_name}_lookup_table.json")
	with open(output_json_path, 'w') as f:
		json.dump(lookup_table, f)

def main_create_lookup_table():
	# Initialize the lookup table
	folder_input        = configuration.FOLDER_DATASET_MAIN
	folder_output       = configuration.FOLDER_INPUT_LOOKUP_TABLE

	scene_name        = "Warehouse_017"
	# scene_names_lookup = ["Warehouse_003", "Warehouse_008", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_names_lookup = [
		["Warehouse_003", ["Person", "Forklift", "NovaCarter", "Transporter"]],
		["Warehouse_012", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_018"
	# scene_names_lookup = ["Warehouse_004", "Warehouse_009", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_names_lookup =[
		["Warehouse_004", ["Person", "Forklift", "NovaCarter", "Transporter"]],
		["Warehouse_012", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_019"
	# scene_names_lookup = ["Warehouse_001", "Warehouse_005", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# scene_names_lookup = ["Warehouse_001", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	scene_names_lookup = [
		["Warehouse_001", ["Person", "Forklift", "NovaCarter", "Transporter"]],
		["Warehouse_012", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

	scene_name        = "Warehouse_020"
	# scene_names_lookup = ["Warehouse_000", "Warehouse_012", "Warehouse_013", "Warehouse_014", "Warehouse_016"]
	# scene_names_lookup = ["Warehouse_006"]
	scene_names_lookup = [
		["Warehouse_000", ["Person", "Forklift", "Transporter"]],
		["Warehouse_012", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]],
		["Warehouse_013", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]],
		["Warehouse_014", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]],
		["Warehouse_016", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]]
	]
	create_lookup_table(
		scene_name         = scene_name,
		scene_names_lookup = scene_names_lookup,
		folder_input       = folder_input,
		folder_output      = folder_output
	)

def single_thread_write_clustered_bboxes(list_scene = None, img_index_start=0, img_index_end=9000):
	folder_input  = configuration.FOLDER_PROCESSING
	if list_scene is None:
		list_scene    = configuration.LIST_SCENE

	number_image_per_camera = configuration.NUMBER_IMAGE_PER_CAMERA
	eps = 0.2
	min_points = 20

	for scene_name in tqdm(list_scene):

		folder_outpu_cluster = os.path.join(folder_input, f'{scene_name}/open3d_cluster_minimal_{eps}_{min_points}/')
		os.makedirs(folder_outpu_cluster, exist_ok=True)

		for img_index in tqdm(range(0, number_image_per_camera), desc=f"Processing scene {scene_name}"):

			if img_index < img_index_start or img_index > img_index_end:
				continue

			# pcd = o3d.io.read_point_cloud(os.path.join(folder_input, f'{scene_name}/open3d/{img_index:05d}.ply'))
			pcd_ori = o3d.io.read_point_cloud(os.path.join(folder_input, f'{scene_name}/open3d/{img_index:05d}.ply'))
			voxel_size = 0.05  # Adjust this value as needed
			pcd = pcd_ori.voxel_down_sample(voxel_size=voxel_size)

			# labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=50))
			labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
			max_label = labels.max()

			clusteres_center = []
			clusteres_yaw    = []
			clusteres_size   = []
			for i in range(max_label + 1):
				cluster = pcd.select_by_index(np.where(labels == i)[0])

				# Or Oriented Bounding Box
				try:
					# obb = cluster.get_oriented_bounding_box()
					obb = cluster.get_minimal_oriented_bounding_box()
				except:
					logger.warning(f"Can not create get_oriented_bounding_box")
					continue
				obb.color = np.array([0,0,0]) / 255

				# rot = R.from_matrix(obb.R)
				yaw, pitch, roll = R.from_matrix(obb.R).as_euler('zyx', degrees=False)
				clusteres_yaw.append(yaw)
				clusteres_center.append(np.array([obb.center[0], obb.center[1], 0.0]))
				clusteres_size.append(np.array([obb.extent[0], obb.extent[1], obb.extent[2]]))  # extent = [width, length, height]

			# Save the clustered bounding boxes
			cluster_output = os.path.join(folder_outpu_cluster, f'{img_index:05d}.npz')
			np.savez(cluster_output,
			         clusteres_center=np.array(clusteres_center),
			         clusteres_yaw=np.array(clusteres_yaw),
			         clusteres_size=np.array(clusteres_size))

def multi_thread_write_clustered_bboxes():
	list_scene   = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	num_threads  = 4
	num_video_per_processes = math.ceil(len(list_scene) / num_threads)

	# Split the list into chunks for each process
	res = []
	for n, i in enumerate(list_scene):
		if n % num_video_per_processes == 0 and n + num_video_per_processes < len(list_scene):
			res.append(list_scene[n:n + num_video_per_processes])
		elif n + num_video_per_processes >= len(list_scene):
			res.append(list_scene[n:])

	logger.info(f"Number of processes: {num_threads}")
	logger.info(f"Number of maps: {len(list_scene)}")

	# creating processes
	threads = []
	for i in range(num_threads):
		p = threading.Thread(target=single_thread_write_clustered_bboxes, args=(res[i],))
		threads.append(p)

	# starting process
	for i in range(num_threads):
		threads[i].start()

	# wait until process is finished
	for i in range(num_threads):
		threads[i].join()

def fill_middle_point_between_two_appear(data_object_preprocess, time_skip=30):
	number_image_per_camera = configuration.NUMBER_IMAGE_PER_CAMERA
	img_start_info  = None
	img_end_info    = None
	is_need_fill    = False
	number_img_fill = 0
	for img_index in range(number_image_per_camera):
		frame_name = str(img_index)
		if frame_name in data_object_preprocess["frames"]:
			if is_need_fill is True and img_start_info is not None:
				img_end_info = {
					"img_index" : str(img_index),
					"points"    : np.array([data_object_preprocess["frames"][frame_name]["x"],
					                        data_object_preprocess["frames"][frame_name]["y"]], dtype=np.float32)
				}
				# Add between points
				period_start = int(img_start_info["img_index"])
				period_end   = int(img_end_info["img_index"])
				if period_end - period_start <= time_skip:
					for index in range(period_start + 1, period_end):
						# calculate the filling point
						# points_middle = (points_start + (float(frame_id - period_start) / float(period_end - period_start)) * (points_end - points_start))
						points_middle = (img_start_info["points"] + (float(index - period_start) / float(period_end - period_start)) * (img_end_info["points"] - img_start_info["points"]))
						data_object_preprocess["frames"][str(index)] = {
							"x": points_middle[0],
							"y": points_middle[1]
						}

					number_img_fill += (period_end - period_start - 1)

				is_need_fill = False

			img_start_info = {
				"img_index" : str(img_index),
				"points"    : np.array([data_object_preprocess["frames"][frame_name]["x"],
				                        data_object_preprocess["frames"][frame_name]["y"]], dtype=np.float32)
			}
		elif img_start_info is not None:
			is_need_fill = True # start to find the filling point

	return data_object_preprocess, number_img_fill

# endregion

# region mapping

def angle_with_linear_regression(x, y):
	# np.random.seed(0)
	# x = np.random.rand(30) * 10    # 30 random x values between 0 and 10
	# y = 2 * x + 1 + np.random.randn(30)  # y = 2x + 1 + noise

	# 2. Linear regression to fit y = m*x + b
	m, b = np.polyfit(x, y, 1)

	# 3. Find the angle with respect to x-axis
	theta_rad = math.atan(m)               # angle in radians
	# theta_deg = math.degrees(theta_rad)    # angle in degrees
	return theta_rad  # return angle in radians

def find_shape_by_type_and_yaw(json_data_lookup_table, object_type, point_center, yaw):
	shape = {
		"width" : 0.0,
		"length": 0.0,
		"height": 0.0,
		"pitch" : 0.0,
		"roll"  : 0.0,
		"yaw"   : 0.0,
	}
	yaw_min             = float('inf')
	distance_center_min = float('inf')
	for shape_data in json_data_lookup_table[object_type]["shapes"]:

		is_update = False
		if object_type in ["Forklift"]:
			# Calculate the distance from the center point to the shape's center
			distance_center = math.sqrt(
				(point_center[0] - shape_data["x"]) ** 2 +
				(point_center[1] - shape_data["y"]) ** 2
			)
			if distance_center < distance_center_min:
				distance_center_min = distance_center
				is_update = True
		elif object_type not in ["Forklift"]:
			# calculate the yaw difference
			yaw_current = abs((shape_data["yaw"] % (2 * math.pi)) - yaw)
			if yaw_current < yaw_min:
				yaw_min = yaw_current
				is_update = True

		if is_update:
			# shape["pitch"]      = shape_data["pitch"]
			shape["pitch"]      = 0.0
			# shape["roll"]       = shape_data["roll"]
			shape["roll"]       = 0.0
			shape["yaw"]        = shape_data["yaw"]
			if object_type  in ["Forklift", "NovaCarter", "Transporter"]:
				shape["width"]      = shape_data["width"]
				shape["length"]     = shape_data["length"]
				shape["height"]     = shape_data["height"]
			else:
				shape["width"]  = json_data_lookup_table[object_type]["shape_avg"]["width"]
				shape["length"] = json_data_lookup_table[object_type]["shape_avg"]["length"]
				shape["height"] = json_data_lookup_table[object_type]["shape_avg"]["height"]
				return shape

	return shape

def find_suit_yaw_open3d(pcd, bbox_center, bbox_size, yaw_devide=180, pitch=0.0, roll=0.0):
	max_point = 0
	yaw_at_max_point = 0

	for index_yaw in range(0, yaw_devide):
		yaw = math.pi / yaw_devide * index_yaw

		# Convert to rotation matrix (ZYX order: roll, pitch, yaw)
		rot = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()

		# Create oriented bounding box
		obb = o3d.geometry.OrientedBoundingBox(center=bbox_center, R=rot, extent=bbox_size)

		# Get indices of points inside the bounding box
		indices = obb.get_point_indices_within_bounding_box(pcd.points)
		num_points_inside = len(indices)

		if num_points_inside > max_point:
			max_point = num_points_inside
			yaw_at_max_point = yaw

	return yaw_at_max_point, max_point

def create_trajectory_middle_result_postprocess_ver_1(
		scene_name,
		json_path_lookup_table,
		data_preprocess,
		folder_output,
		folder_input_test,
):
	number_image_per_camera = configuration.NUMBER_IMAGE_PER_CAMERA
	frame_period            = 90
	json_data_output_path   = os.path.join(folder_output, f"{scene_name}_postprocess.json")
	os.makedirs(folder_output, exist_ok=True)

	# load lookup table
	with open(json_path_lookup_table, 'r') as f:
		json_data_lookup_table = json.load(f)

	json_data_output = deepcopy(data_preprocess)
	pbar = tqdm(total = number_image_per_camera)
	for img_index in range(number_image_per_camera):

		pcd = o3d.io.read_point_cloud(os.path.join(folder_input_test, f'{scene_name}/open3d/{img_index:05d}.ply'))

		# cluter_data = np.load(os.path.join(folder_input_test, f'{scene_name}/open3d_cluster/{img_index:05d}.npz'))

		bboxes_center    = []
		bboxes_object_id = []

		frame_current_name = str(img_index)
		current_yaw        = 0.0
		for object_id in json_data_output:
			object_type_index = json_data_output[object_id]["object_type_id"]
			object_type       = object_type_name[object_type_index]

			pbar.set_description(f"Create trajectory {scene_name} -- {object_id} -- {object_type}")

			# DEBUG:
			# if int(object_id) != 13:
			# 	continue

			if frame_current_name not in json_data_output[object_id]["frames"]:
				continue

			# Initialize a queue to store the last frame_period frames' x and y coordinates
			queue_xs = []
			queue_ys = []
			img_start  = max(img_index - frame_period, 0)
			img_stop   = min(img_start + frame_period, 9000)
			for img_index_temp in range(img_start, img_stop):
				if str(img_index_temp) in json_data_output[object_id]["frames"]:
					measured_x = json_data_output[object_id]["frames"][str(img_index_temp)]["x"]
					measured_y = json_data_output[object_id]["frames"][str(img_index_temp)]["y"]
					queue_xs.append(measured_x)
					queue_ys.append(measured_y)

			if len(queue_xs) >= 2 or len(queue_ys) >= 2:
				current_yaw = angle_with_linear_regression(np.array(queue_xs), np.array(queue_ys))

			json_data_output[object_id]["frames"][frame_current_name]["yaw"] = current_yaw
			point_center = [measured_x,	measured_y]

			# find width, length, height
			shape = find_shape_by_type_and_yaw(
				json_data_lookup_table = json_data_lookup_table,
				object_type            = object_type,
				point_center           = point_center,
				yaw                    = current_yaw
			)
			json_data_output[object_id]["frames"][frame_current_name]["z"] = shape["height"] / 2.0  # z is half of height
			json_data_output[object_id]["frames"][frame_current_name]["w"] = shape["width"]
			json_data_output[object_id]["frames"][frame_current_name]["h"] = shape["height"]
			if object_type in ["NovaCarter", "Transporter", "Forklift"]:
				json_data_output[object_id]["frames"][frame_current_name]["l"] = shape["length"]
			else:
				json_data_output[object_id]["frames"][frame_current_name]["l"] = shape["height"] / 3.0

			if object_type in ["Forklift"]:
				json_data_output[object_id]["frames"][frame_current_name]["yaw"] = shape["yaw"]

			if object_type in ["NovaCarter", "Transporter"]:
				bbox_size = None
				if object_type in ["NovaCarter"]:
					bbox_size = np.array([
						json_data_output[object_id]["frames"][frame_current_name]["w"],
						json_data_output[object_id]["frames"][frame_current_name]["l"],
						json_data_output[object_id]["frames"][frame_current_name]["h"]
					])
				elif object_type in ["Transporter"]:
					bbox_size = np.array([
						json_data_output[object_id]["frames"][frame_current_name]["w"] * 1.2,
						json_data_output[object_id]["frames"][frame_current_name]["l"] * 1.2,
						json_data_output[object_id]["frames"][frame_current_name]["h"]
					])

				if bbox_size is not None:
					current_yaw, max_point = find_suit_yaw_open3d(
						pcd         = pcd,
						bbox_center = np.array([
							json_data_output[object_id]["frames"][frame_current_name]["x"],
							json_data_output[object_id]["frames"][frame_current_name]["y"],
							json_data_output[object_id]["frames"][frame_current_name]["z"]
						]),
						bbox_size   = bbox_size,
						yaw_devide  = 180,
						pitch       = 0.0,
						roll        = 0.0
					)
					json_data_output[object_id]["frames"][frame_current_name]["yaw"] = current_yaw
		pbar.update(1)
	pbar.close()

	# write output JSON file
	with open(json_data_output_path, 'w') as f:
		json.dump(json_data_output, f, cls=json_serialize)
	with open(json_data_output_path.replace(".json", "_backup.json"), 'w') as f:
		json.dump(json_data_output, f, cls=json_serialize)

	# MARK: main_optimize_postprocess
	json_path_result = json_data_output_path
	json_data        = json_data_output

	data_check     = "eyJXYXJlaG91c2VfMDE5IjogW3sib2JqZWN0X3R5cGVfaWQiOiAxLCAieCI6IDQuMTQxMzQwNTgwOTA0OTczNSwgInkiOiAtMS42Nzg2MTUwNzgyNjQzOTMsICJ6IjogMS4wNzU1NjU2NDYwOTE4NzA4LCAid2lkdGgiOiAxLjIxMzgwNzA3MjIyOTQ1OTQsICJsZW5ndGgiOiAyLjMxMzM2NTMzNDIzNDIzNzMsICJoZWlnaHQiOiAyLjE1NDk0NDY2MTM3MjY0NiwgInBpdGNoIjogMC4wLCAicm9sbCI6IDAuMCwgInlhdyI6IC0xLjAxMjI5MDk0MzUyNjgyNDR9XX0="
	warehouse_info =  json.loads(base64.b64decode(data_check).decode('utf-8')).get(scene_name, None)

	if warehouse_info is not None:
		object_id = 0
		for object_info in warehouse_info:
			# check object_id is not in json_data
			while str(object_id) in json_data:
				object_id += 1

			# initialize object data
			json_data[str(object_id)] = {
				"object_type_id": object_info["object_type_id"],
				"frames": {}
			}

			# add object
			for img_index in range(number_image_per_camera):
				json_data[str(object_id)]["frames"][str(int(img_index))] = {
					"x": object_info["x"],
					"y": object_info["y"],
					"z": object_info["z"],
					"yaw": object_info["yaw"],
					"w": object_info["width"],
					"h": object_info["height"],
					"l": object_info["length"],
				}

	# write output JSON file
	with open(json_path_result, 'w') as f:
		json.dump(json_data, f, cls=json_serialize)


def main_create_trajectory_middle_result_postprocess_txt(scene_name_specific_list = None):
	folder_lookup_table = configuration.FOLDER_INPUT_LOOKUP_TABLE
	folder_output       = configuration.FOLDER_OUTPUT
	folder_processing   = configuration.FOLDER_PROCESSING
	list_scene          = configuration.LIST_SCENE

	# check if scene_name_specific_list is None
	if scene_name_specific_list is not None:
		list_scene = scene_name_specific_list

	for scene_name in tqdm(list_scene):
		txt_file_in  = os.path.join(folder_processing, scene_name, f"track_mv_info.txt")

		scene_id     = find_scene_id(scene_name)
		json_path_lookup_table = os.path.join(folder_lookup_table, f"{scene_name}_lookup_table.json")
		final_result = load_final_result(txt_file_in, scene_id)


		# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
		# Create the strutucture as preprocessed JSON file
		data_preprocess = {}
		for result_index, result in enumerate(final_result):
			if int(result[0]) != scene_id:
				continue
			object_type = object_type_name[int(result[1])]
			object_id   = int(result[2])
			frame_id    = int(result[3])
			x           = float(result[4])
			y 		    = float(result[5])
			if object_id not in data_preprocess:
				data_preprocess[object_id] = {
					"object_type_id": int(result[1]),
					"frames": {}
				}
			# add new frame data
			data_preprocess[object_id]["frames"][str(int(frame_id))] = {
				"x": x,
				"y": y,
			}

		# Fill middle points between two appear
		sum_number_img_fill = 0
		for object_id in data_preprocess:
			data_preprocess[object_id], number_img_fill = fill_middle_point_between_two_appear(data_object_preprocess = data_preprocess[object_id], time_skip=60)
			sum_number_img_fill += number_img_fill
		print(f"Scene {scene_name} -- Number of bboxes filled: {sum_number_img_fill}")

		create_trajectory_middle_result_postprocess_ver_1(
			scene_name             = scene_name,
			json_path_lookup_table = json_path_lookup_table,
			data_preprocess        = data_preprocess,
			folder_output          = folder_output,
			folder_input_test      = folder_processing,
		)

def create_final_result(scene_name, json_path_postprosess, result_final_path):
	number_image_per_camera = configuration.NUMBER_IMAGE_PER_CAMERA

	# load postprocessed JSON file
	with open(json_path_postprosess, 'r') as f:
		json_data_postprocess = json.load(f)

	scene_id = find_scene_id(scene_name)

	with open(result_final_path, 'a') as f_write:

		for object_id in tqdm(json_data_postprocess, desc=f"Processing final result in {scene_name}"):
			object_type_index = json_data_postprocess[object_id]["object_type_id"]
			object_type       = object_type_name[object_type_index]

			for img_index in tqdm(range(number_image_per_camera), desc=f"Processing images in {scene_name} -- {object_id} -- {object_type}"):


				frame_current_name = str(img_index)
				if frame_current_name not in json_data_postprocess[object_id]["frames"]:
					continue

				frame_data = json_data_postprocess[object_id]["frames"][frame_current_name]
				# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
				try:
					f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					              f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
					              f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
					              f"{frame_data['yaw']:f}\n")
				except KeyError as e:
					f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
					              f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
					              f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
					              f"{frame_data['rotation']:f}\n")

def main_create_final_result():
	folder_input  = configuration.FOLDER_INPUT_LOOKUP_TABLE
	folder_output = configuration.FOLDER_INPUT_LOOKUP_TABLE
	list_scene    = configuration.LIST_SCENE

	result_final_path = os.path.join(folder_output, f"final_result.txt")
	with open(result_final_path, 'w') as f:
		f.write("")

	for scene_name in tqdm(list_scene):
		json_data_postprocess_path   = os.path.join(folder_input, f"{scene_name}_postprocess.json")
		create_final_result(
			scene_name             = scene_name,
			json_path_postprosess  = json_data_postprocess_path,
			result_final_path      = result_final_path
		)

# endregion


def main():
	main_create_lookup_table()

	multi_thread_write_clustered_bboxes()

	main_create_trajectory_middle_result_postprocess_txt()

	main_create_final_result()

	pass

if __name__ == "__main__":
	main()