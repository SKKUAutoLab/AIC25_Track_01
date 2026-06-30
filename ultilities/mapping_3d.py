import glob
import json
import math
import os
import threading
from copy import deepcopy
from dataclasses import dataclass
from functools import cmp_to_key

import numpy as np
import open3d as o3d
from loguru import logger
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ultilities import configuration


CLUSTER_SCENES = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

LOOKUP_TABLE_PLAN = [
	("Warehouse_017", [
		("Warehouse_003", ["Person", "Forklift", "NovaCarter", "Transporter"]),
		("Warehouse_012", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_013", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_014", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_016", ["AgilityDigit", "FourierGR1T2"]),
	]),
	("Warehouse_018", [
		("Warehouse_004", ["Person", "Forklift", "NovaCarter", "Transporter"]),
		("Warehouse_012", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_013", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_014", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_016", ["AgilityDigit", "FourierGR1T2"]),
	]),
	("Warehouse_019", [
		("Warehouse_001", ["Person", "Forklift", "NovaCarter", "Transporter"]),
		("Warehouse_012", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_013", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_014", ["AgilityDigit", "FourierGR1T2"]),
		("Warehouse_016", ["AgilityDigit", "FourierGR1T2"]),
	]),
	("Warehouse_020", [
		("Warehouse_000", ["Person", "Forklift", "Transporter"]),
		("Warehouse_012", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]),
		("Warehouse_013", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]),
		("Warehouse_014", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]),
		("Warehouse_016", ["NovaCarter", "AgilityDigit", "FourierGR1T2"]),
	]),
]


class NumpyJsonEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)


class SceneRegistry:
	_SCENE_NAMES_BY_SPLIT = {
		"Train": {
			0: "Warehouse_000",
			1: "Warehouse_001",
			2: "Warehouse_002",
			3: "Warehouse_003",
			4: "Warehouse_004",
			5: "Warehouse_005",
			6: "Warehouse_006",
			7: "Warehouse_007",
			8: "Warehouse_008",
			9: "Warehouse_009",
			10: "Warehouse_010",
			11: "Warehouse_011",
			12: "Warehouse_012",
			13: "Warehouse_013",
			14: "Warehouse_014",
		},
		"Val": {
			15: "Warehouse_015",
			16: "Warehouse_016",
			22: "Lab_000",
			23: "Hospital_000",
		},
		"Test": {
			17: "Warehouse_017",
			18: "Warehouse_018",
			19: "Warehouse_019",
			20: "Warehouse_020",
		},
	}

	def __init__(self):
		self._id_by_name = {
			name: scene_id
			for split in self._SCENE_NAMES_BY_SPLIT.values()
			for scene_id, name in split.items()
		}

	def scene_id(self, scene_name):
		scene_id = self._id_by_name.get(scene_name)
		if scene_id is None:
			logger.error(f"Scene name {scene_name} not found in any dataset split.")
		return scene_id


class ObjectTypeCatalog:
	_NAME_BY_ID = {
		0: "Person",
		1: "Forklift",
		2: "NovaCarter",
		3: "Transporter",
		4: "FourierGR1T2",
		5: "AgilityDigit",
	}
	_COLOR_BY_NAME = {
		"Person": (77, 109, 163),
		"Forklift": (162, 245, 214),
		"NovaCarter": (245, 245, 245),
		"Transporter": (0, 255, 255),
		"FourierGR1T2": (164, 17, 157),
		"AgilityDigit": (235, 229, 52),
	}

	def items(self):
		return self._NAME_BY_ID.items()

	def name(self, type_id):
		return self._NAME_BY_ID[type_id]

	def color(self, type_name):
		return self._COLOR_BY_NAME[type_name]


@dataclass
class Shape:
	width: float = 0.0
	length: float = 0.0
	height: float = 0.0
	pitch: float = 0.0
	roll: float = 0.0
	yaw: float = 0.0


class ShapeLookup:
	def __init__(self, table):
		self._table = table

	def find(self, object_type, point_center, yaw):
		shape = Shape()
		yaw_min = float('inf')
		distance_center_min = float('inf')
		for sample in self._table[object_type]["shapes"]:
			is_update = False
			if object_type == "Forklift":
				distance_center = math.sqrt(
					(point_center[0] - sample["x"]) ** 2 +
					(point_center[1] - sample["y"]) ** 2
				)
				if distance_center < distance_center_min:
					distance_center_min = distance_center
					is_update = True
			else:
				yaw_current = abs((sample["yaw"] % (2 * math.pi)) - yaw)
				if yaw_current < yaw_min:
					yaw_min = yaw_current
					is_update = True

			if not is_update:
				continue

			shape.pitch = 0.0
			shape.roll = 0.0
			shape.yaw = sample["yaw"]
			if object_type in ("Forklift", "NovaCarter", "Transporter"):
				shape.width = sample["width"]
				shape.length = sample["length"]
				shape.height = sample["height"]
			else:
				average = self._table[object_type]["shape_avg"]
				shape.width = average["width"]
				shape.length = average["length"]
				shape.height = average["height"]
				return shape

		return shape


def load_track_results(track_path, scene_id):
	def compare(part_a, part_b):
		if part_a[3] != part_b[3]:
			return int(part_a[3]) - int(part_b[3])
		if part_a[2] != part_b[2]:
			return int(part_a[2]) - int(part_b[2])
		if part_a[1] != part_b[1]:
			return int(part_a[1]) - int(part_b[1])
		return 0

	results = []
	with open(track_path, 'r') as f:
		for line in f.readlines():
			if line.startswith("#") or line.strip() == "":
				continue
			separator = "," if "," in line else None
			parts = np.array(line.split(separator), dtype=np.float32)
			if int(parts[0]) == scene_id:
				results.append(parts)

	return sorted(results, key=cmp_to_key(compare))


def yaw_from_linear_regression(xs, ys):
	slope, _ = np.polyfit(xs, ys, 1)
	return math.atan(slope)


def best_yaw_by_point_density(pcd, bbox_center, bbox_size, yaw_divisions=180, pitch=0.0, roll=0.0):
	max_point = 0
	yaw_at_max_point = 0

	for index_yaw in range(0, yaw_divisions):
		yaw = math.pi / yaw_divisions * index_yaw
		rotation = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
		obb = o3d.geometry.OrientedBoundingBox(center=bbox_center, R=rotation, extent=bbox_size)
		num_points_inside = len(obb.get_point_indices_within_bounding_box(pcd.points))
		if num_points_inside > max_point:
			max_point = num_points_inside
			yaw_at_max_point = yaw

	return yaw_at_max_point, max_point


def fill_missing_track_points(object_data, number_image_per_camera, time_skip=30):
	start_info = None
	is_need_fill = False
	number_img_fill = 0
	for img_index in range(number_image_per_camera):
		frame_name = str(img_index)
		if frame_name in object_data["frames"]:
			if is_need_fill is True and start_info is not None:
				end_info = {
					"img_index": str(img_index),
					"points": np.array([object_data["frames"][frame_name]["x"],
					                    object_data["frames"][frame_name]["y"]], dtype=np.float32),
				}
				period_start = int(start_info["img_index"])
				period_end = int(end_info["img_index"])
				if period_end - period_start <= time_skip:
					for index in range(period_start + 1, period_end):
						points_middle = (start_info["points"]
						                 + (float(index - period_start) / float(period_end - period_start))
						                 * (end_info["points"] - start_info["points"]))
						object_data["frames"][str(index)] = {
							"x": points_middle[0],
							"y": points_middle[1],
						}
					number_img_fill += (period_end - period_start - 1)

				is_need_fill = False

			start_info = {
				"img_index": str(img_index),
				"points": np.array([object_data["frames"][frame_name]["x"],
				                    object_data["frames"][frame_name]["y"]], dtype=np.float32),
			}
		elif start_info is not None:
			is_need_fill = True

	return object_data, number_img_fill


class LookupTableBuilder:
	def __init__(self, catalog, folder_input, folder_output):
		self._catalog = catalog
		self._folder_input = folder_input
		self._folder_output = folder_output

	def build(self, scene_name, scene_specs):
		os.makedirs(self._folder_output, exist_ok=True)
		table = self._empty_table()
		for scene, allowed_types in tqdm(scene_specs, desc=f"Creating lookup table for {scene_name}"):
			self._accumulate_scene(table, scene, allowed_types)
		self._finalize_averages(table)
		self._log_statistics(table)
		self._save(table, scene_name)

	def _empty_table(self):
		table = {}
		for type_id, type_name in self._catalog.items():
			table[type_name] = {
				"type_id": type_id,
				"color": self._catalog.color(type_name),
				"shape_max": {"width": 0, "length": 0, "height": 0},
				"shape_min": {"width": float('inf'), "length": float('inf'), "height": float('inf')},
				"shape_avg": {"width": 0.0, "length": 0.0, "height": 0.0},
				"shape_count": 0,
				"shapes": [],
			}
		return table

	def _accumulate_scene(self, table, scene, allowed_types):
		groundtruth_path = glob.glob(os.path.join(self._folder_input, f"*/{scene}/ground_truth.json"))[0]
		with open(groundtruth_path, 'r') as f:
			groundtruth = json.load(f)

		for _, frame_data in tqdm(groundtruth.items(), desc=f"Processing creating lookup table in {scene}"):
			for instance in frame_data:
				object_type = instance["object type"]
				if object_type not in allowed_types:
					continue
				self._accumulate_instance(table[object_type], instance)

	@staticmethod
	def _accumulate_instance(entry, instance):
		location = instance["3d location"]
		scale = instance["3d bounding box scale"]
		rotation = instance["3d bounding box rotation"]

		entry["shapes"].append({
			"x": location[0],
			"y": location[1],
			"z": location[2],
			"width": scale[0],
			"length": scale[1],
			"height": scale[2],
			"pitch": rotation[0],
			"roll": rotation[1],
			"yaw": rotation[2],
		})

		entry["shape_max"]["width"] = max(entry["shape_max"]["width"], scale[0])
		entry["shape_min"]["width"] = min(entry["shape_min"]["width"], scale[0])
		entry["shape_avg"]["width"] += scale[0]
		entry["shape_max"]["length"] = max(entry["shape_max"]["length"], scale[1])
		entry["shape_min"]["length"] = min(entry["shape_min"]["length"], scale[1])
		entry["shape_avg"]["length"] += scale[1]
		entry["shape_max"]["height"] = max(entry["shape_max"]["height"], scale[2])
		entry["shape_min"]["height"] = min(entry["shape_min"]["height"], scale[2])
		entry["shape_avg"]["height"] += scale[2]
		entry["shape_count"] += 1

	def _finalize_averages(self, table):
		for _, type_name in self._catalog.items():
			entry = table[type_name]
			if entry["shape_count"] > 0:
				entry["shape_avg"]["width"] /= entry["shape_count"]
				entry["shape_avg"]["length"] /= entry["shape_count"]
				entry["shape_avg"]["height"] /= entry["shape_count"]

	def _log_statistics(self, table):
		for _, type_name in self._catalog.items():
			entry = table[type_name]
			print(f"{type_name} shape ::::: \n "
			      f"max -- {entry['shape_max']}, \n "
			      f"min -- {entry['shape_min']}, \n "
			      f"avg -- {entry['shape_avg']}, \n "
			      f"count -- {entry['shape_count']}")

	def _save(self, table, scene_name):
		output_path = os.path.join(self._folder_output, f"{scene_name}_lookup_table.json")
		with open(output_path, 'w') as f:
			json.dump(table, f)


class ClusterWriter:
	def __init__(self, folder_input, number_image_per_camera, eps=0.2, min_points=20, voxel_size=0.05):
		self._folder_input = folder_input
		self._number_image_per_camera = number_image_per_camera
		self._eps = eps
		self._min_points = min_points
		self._voxel_size = voxel_size

	def write_scenes(self, list_scene, img_index_start=0, img_index_end=9000):
		for scene_name in tqdm(list_scene):
			self.write_scene(scene_name, img_index_start, img_index_end)

	def write_scene(self, scene_name, img_index_start=0, img_index_end=9000):
		output_folder = os.path.join(
			self._folder_input, f'{scene_name}/open3d_cluster_minimal_{self._eps}_{self._min_points}/')
		os.makedirs(output_folder, exist_ok=True)
		for img_index in tqdm(range(0, self._number_image_per_camera), desc=f"Processing scene {scene_name}"):
			if img_index < img_index_start or img_index > img_index_end:
				continue
			self._write_frame(scene_name, output_folder, img_index)

	def _write_frame(self, scene_name, output_folder, img_index):
		pcd_ori = o3d.io.read_point_cloud(
			os.path.join(self._folder_input, f'{scene_name}/open3d/{img_index:05d}.ply'))
		pcd = pcd_ori.voxel_down_sample(voxel_size=self._voxel_size)
		labels = np.array(pcd.cluster_dbscan(eps=self._eps, min_points=self._min_points))
		max_label = labels.max()

		centers = []
		yaws = []
		sizes = []
		for i in range(max_label + 1):
			cluster = pcd.select_by_index(np.where(labels == i)[0])
			try:
				obb = cluster.get_minimal_oriented_bounding_box()
			except Exception:
				logger.warning(f"Can not create get_oriented_bounding_box")
				continue
			obb.color = np.array([0, 0, 0]) / 255

			yaw, pitch, roll = R.from_matrix(obb.R).as_euler('zyx', degrees=False)
			yaws.append(yaw)
			centers.append(np.array([obb.center[0], obb.center[1], 0.0]))
			sizes.append(np.array([obb.extent[0], obb.extent[1], obb.extent[2]]))

		output_path = os.path.join(output_folder, f'{img_index:05d}.npz')
		np.savez(output_path,
		         clusteres_center=np.array(centers),
		         clusteres_yaw=np.array(yaws),
		         clusteres_size=np.array(sizes))

	def write_scenes_threaded(self, list_scene, num_threads=4):
		chunks = self._split_scenes(list_scene, num_threads)
		logger.info(f"Number of processes: {num_threads}")
		logger.info(f"Number of maps: {len(list_scene)}")

		threads = [threading.Thread(target=self.write_scenes, args=(chunk,)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

	@staticmethod
	def _split_scenes(list_scene, num_threads):
		num_per_thread = math.ceil(len(list_scene) / num_threads)
		chunks = []
		for n, _ in enumerate(list_scene):
			if n % num_per_thread == 0 and n + num_per_thread < len(list_scene):
				chunks.append(list_scene[n:n + num_per_thread])
			elif n + num_per_thread >= len(list_scene):
				chunks.append(list_scene[n:])
		return chunks


class TrajectoryStage:
	def __init__(self, catalog, scene_registry, number_image_per_camera,
	             folder_lookup_table, folder_processing, folder_output):
		self._catalog = catalog
		self._scene_registry = scene_registry
		self._number_image_per_camera = number_image_per_camera
		self._folder_lookup_table = folder_lookup_table
		self._folder_processing = folder_processing
		self._folder_output = folder_output

	def run(self, list_scene):
		for scene_name in tqdm(list_scene):
			self._run_scene(scene_name)

	def _run_scene(self, scene_name):
		track_path = os.path.join(self._folder_processing, scene_name, f"track_mv_info.txt")
		lookup_path = os.path.join(self._folder_lookup_table, f"{scene_name}_lookup_table.json")
		scene_id = self._scene_registry.scene_id(scene_name)

		results = load_track_results(track_path, scene_id)
		data_preprocess = self._build_preprocess(results, scene_id)
		self._fill_missing(scene_name, data_preprocess)
		self._build_trajectory(scene_name, lookup_path, data_preprocess)

	@staticmethod
	def _build_preprocess(results, scene_id):
		data_preprocess = {}
		for result in results:
			if int(result[0]) != scene_id:
				continue
			object_id = int(result[2])
			frame_id = int(result[3])
			if object_id not in data_preprocess:
				data_preprocess[object_id] = {
					"object_type_id": int(result[1]),
					"frames": {},
				}
			data_preprocess[object_id]["frames"][str(int(frame_id))] = {
				"x": float(result[4]),
				"y": float(result[5]),
			}
		return data_preprocess

	def _fill_missing(self, scene_name, data_preprocess):
		total_filled = 0
		for object_id in data_preprocess:
			data_preprocess[object_id], number_img_fill = fill_missing_track_points(
				object_data=data_preprocess[object_id],
				number_image_per_camera=self._number_image_per_camera,
				time_skip=60,
			)
			total_filled += number_img_fill
		print(f"Scene {scene_name} -- Number of bboxes filled: {total_filled}")

	def _build_trajectory(self, scene_name, lookup_path, data_preprocess):
		os.makedirs(self._folder_output, exist_ok=True)
		output_path = os.path.join(self._folder_output, f"{scene_name}_postprocess.json")

		with open(lookup_path, 'r') as f:
			shape_lookup = ShapeLookup(json.load(f))

		json_data_output = deepcopy(data_preprocess)
		pbar = tqdm(total=self._number_image_per_camera)
		for img_index in range(self._number_image_per_camera):
			pcd = o3d.io.read_point_cloud(
				os.path.join(self._folder_processing, f'{scene_name}/open3d/{img_index:05d}.ply'))

			frame_current_name = str(img_index)
			current_yaw = 0.0
			for object_id in json_data_output:
				object_type_index = json_data_output[object_id]["object_type_id"]
				object_type = self._catalog.name(object_type_index)
				pbar.set_description(f"Create trajectory {scene_name} -- {object_id} -- {object_type}")

				if frame_current_name not in json_data_output[object_id]["frames"]:
					continue

				queue_xs = []
				queue_ys = []
				img_start = max(img_index - 90, 0)
				img_stop = min(img_start + 90, 9000)
				for img_index_temp in range(img_start, img_stop):
					if str(img_index_temp) in json_data_output[object_id]["frames"]:
						measured_x = json_data_output[object_id]["frames"][str(img_index_temp)]["x"]
						measured_y = json_data_output[object_id]["frames"][str(img_index_temp)]["y"]
						queue_xs.append(measured_x)
						queue_ys.append(measured_y)

				if len(queue_xs) >= 2 or len(queue_ys) >= 2:
					current_yaw = yaw_from_linear_regression(np.array(queue_xs), np.array(queue_ys))

				frame = json_data_output[object_id]["frames"][frame_current_name]
				frame["yaw"] = current_yaw
				point_center = [measured_x, measured_y]

				shape = shape_lookup.find(object_type, point_center, current_yaw)
				frame["z"] = shape.height / 2.0
				frame["w"] = shape.width
				frame["h"] = shape.height
				if object_type in ("NovaCarter", "Transporter", "Forklift"):
					frame["l"] = shape.length
				else:
					frame["l"] = shape.height / 3.0

				if object_type == "Forklift":
					frame["yaw"] = shape.yaw

				if object_type in ("NovaCarter", "Transporter"):
					bbox_size = self._bbox_size(object_type, frame)
					if bbox_size is not None:
						current_yaw, _ = best_yaw_by_point_density(
							pcd=pcd,
							bbox_center=np.array([frame["x"], frame["y"], frame["z"]]),
							bbox_size=bbox_size,
							yaw_divisions=180,
							pitch=0.0,
							roll=0.0,
						)
						frame["yaw"] = current_yaw
			pbar.update(1)
		pbar.close()

		with open(output_path, 'w') as f:
			json.dump(json_data_output, f, cls=NumpyJsonEncoder)
		with open(output_path.replace(".json", "_backup.json"), 'w') as f:
			json.dump(json_data_output, f, cls=NumpyJsonEncoder)

	@staticmethod
	def _bbox_size(object_type, frame):
		if object_type == "NovaCarter":
			return np.array([frame["w"], frame["l"], frame["h"]])
		if object_type == "Transporter":
			return np.array([frame["w"] * 1.2, frame["l"] * 1.2, frame["h"]])
		return None


class FinalResultWriter:
	def __init__(self, catalog, scene_registry, number_image_per_camera):
		self._catalog = catalog
		self._scene_registry = scene_registry
		self._number_image_per_camera = number_image_per_camera

	def run(self, list_scene, folder_input, result_path):
		with open(result_path, 'w') as f:
			f.write("")
		for scene_name in tqdm(list_scene):
			postprocess_path = os.path.join(folder_input, f"{scene_name}_postprocess.json")
			self.write_scene(scene_name, postprocess_path, result_path)

	def write_scene(self, scene_name, postprocess_path, result_path):
		with open(postprocess_path, 'r') as f:
			postprocess = json.load(f)
		scene_id = self._scene_registry.scene_id(scene_name)

		with open(result_path, 'a') as f_write:
			for object_id in tqdm(postprocess, desc=f"Processing final result in {scene_name}"):
				self._write_object(f_write, scene_name, scene_id, object_id, postprocess[object_id])

	def _write_object(self, f_write, scene_name, scene_id, object_id, object_data):
		object_type_index = object_data["object_type_id"]
		object_type = self._catalog.name(object_type_index)
		for img_index in tqdm(range(self._number_image_per_camera),
		                      desc=f"Processing images in {scene_name} -- {object_id} -- {object_type}"):
			frame_current_name = str(img_index)
			if frame_current_name not in object_data["frames"]:
				continue
			frame_data = object_data["frames"][frame_current_name]
			self._write_frame(f_write, scene_id, object_type_index, object_id, img_index, frame_data)

	@staticmethod
	def _write_frame(f_write, scene_id, object_type_index, object_id, img_index, frame_data):
		try:
			yaw = frame_data['yaw']
		except KeyError:
			yaw = frame_data['rotation']
		f_write.write(f"{scene_id} {object_type_index} {int(object_id)} {img_index} "
		              f"{frame_data['x']:f} {frame_data['y']:f} {frame_data['z']:f} "
		              f"{frame_data['w']:f} {frame_data['l']:f} {frame_data['h']:f} "
		              f"{yaw:f}\n")


def main():
	catalog = ObjectTypeCatalog()
	scene_registry = SceneRegistry()
	number_image_per_camera = configuration.NUMBER_IMAGE_PER_CAMERA

	lookup_builder = LookupTableBuilder(
		catalog=catalog,
		folder_input=configuration.FOLDER_DATASET_MAIN,
		folder_output=configuration.FOLDER_INPUT_LOOKUP_TABLE,
	)
	for scene_name, scene_specs in LOOKUP_TABLE_PLAN:
		lookup_builder.build(scene_name, scene_specs)

	cluster_writer = ClusterWriter(
		folder_input=configuration.FOLDER_PROCESSING,
		number_image_per_camera=number_image_per_camera,
	)
	cluster_writer.write_scenes_threaded(CLUSTER_SCENES, num_threads=4)

	trajectory_stage = TrajectoryStage(
		catalog=catalog,
		scene_registry=scene_registry,
		number_image_per_camera=number_image_per_camera,
		folder_lookup_table=configuration.FOLDER_INPUT_LOOKUP_TABLE,
		folder_processing=configuration.FOLDER_PROCESSING,
		folder_output=configuration.FOLDER_OUTPUT,
	)
	trajectory_stage.run(configuration.LIST_SCENE)

	final_writer = FinalResultWriter(
		catalog=catalog,
		scene_registry=scene_registry,
		number_image_per_camera=number_image_per_camera,
	)
	result_path = os.path.join(configuration.FOLDER_INPUT_LOOKUP_TABLE, f"final_result.txt")
	final_writer.run(
		list_scene=configuration.LIST_SCENE,
		folder_input=configuration.FOLDER_INPUT_LOOKUP_TABLE,
		result_path=result_path,
	)


if __name__ == "__main__":
	main()
