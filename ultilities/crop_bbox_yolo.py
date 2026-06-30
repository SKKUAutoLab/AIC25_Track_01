import argparse
import os
import glob
import json
import shutil
from dataclasses import dataclass

import cv2
from tqdm import tqdm
from loguru import logger

from mtmc.core.objects.units import Camera
from ultilities import configuration


class ObjectTypeCatalog:
	_NAME_BY_ID = {
		0: "Person",
		1: "Forklift",
		2: "NovaCarter",
		3: "Transporter",
		4: "FourierGR1T2",
		5: "AgilityDigit",
	}

	def name(self, class_id):
		return self._NAME_BY_ID[class_id]

	def is_person(self, class_id):
		return self.name(class_id) == "Person"


@dataclass
class Detection:
	class_id: int
	x_center: float
	y_center: float
	width: float
	height: float
	score: float

	@classmethod
	def parse(cls, line):
		components = line.strip().split()
		if len(components) < 5:
			return None
		return cls(
			int(components[0]),
			float(components[1]),
			float(components[2]),
			float(components[3]),
			float(components[4]),
			float(components[5]),
		)

	def is_too_small(self, img_w, img_h):
		return self.width * img_w < 10 or self.height * img_h < 10

	def pixel_box(self, img_w, img_h):
		x_min = int((self.x_center - self.width / 2) * img_w)
		y_min = int((self.y_center - self.height / 2) * img_h)
		x_max = int((self.x_center + self.width / 2) * img_w)
		y_max = int((self.y_center + self.height / 2) * img_h)
		return x_min, y_min, x_max, y_max


def label_index_sort(file_path):
	stem = os.path.splitext(os.path.basename(file_path))[0]
	return int(stem.split("_")[-1])


class LabelRenamer:
	def __init__(self, image_result_test):
		self.image_result_test = image_result_test

	def rename_scene(self, scene_name):
		folder_input  = os.path.join(self.image_result_test, scene_name, "detection/labels/")
		folder_output = os.path.join(self.image_result_test, scene_name, "detection/labels_renamed/")
		os.makedirs(folder_output, exist_ok=True)

		list_txt = glob.glob(os.path.join(folder_input, "*.txt"))
		for txt_path in tqdm(list_txt, desc=f"Rename file in scene name {scene_name} : "):
			self._copy_renamed(txt_path, folder_output)

	@staticmethod
	def _copy_renamed(txt_path, folder_output):
		new_index = int(os.path.basename(txt_path).split("_")[-1].split(".")[0]) - 1
		target = os.path.join(folder_output, f'{new_index:08d}.txt')
		try:
			shutil.copy(txt_path, target)
		except FileNotFoundError:
			pass
		except OSError:
			pass


class BboxCropper:
	def __init__(self, image_result_test):
		self.image_result_test = image_result_test

	def crop_scene(self, scene_name):
		scene_folder          = os.path.join(self.image_result_test, scene_name)
		folder_input_lbl      = os.path.join(scene_folder, "detection/labels_renamed")
		folder_output_cropped = os.path.join(scene_folder, "image_croped_test")
		os.makedirs(folder_output_cropped, exist_ok=True)

		list_lbl = sorted(glob.glob(os.path.join(folder_input_lbl, "*.txt")), key=label_index_sort)
		object_index = 0
		pbar = tqdm(total=len(list_lbl))
		for lbl_path in list_lbl:
			image_index = int(os.path.splitext(os.path.basename(lbl_path))[0])
			frame_id    = image_index % 9000
			camera_name = Camera.adjust_camera_id(str(image_index // 9000))

			pbar.set_description(f"Processing croping {scene_name} -- {camera_name} -- {frame_id}")

			img_path = os.path.join(configuration.FOLDER_INPUT_FULL_EXTRACTION_IMAGE, scene_name, camera_name, f"{frame_id:08d}.jpg")
			image = self._load_image(img_path, lbl_path)
			if image is None:
				continue

			object_index = self._crop_labels(lbl_path, image, camera_name, scene_name, frame_id, folder_output_cropped, object_index)
			pbar.update(1)
		pbar.close()

	@staticmethod
	def _load_image(img_path, lbl_path):
		if not os.path.exists(img_path):
			logger.warning(f"Image not found for label: {lbl_path}")
			logger.warning(f"Expected image path: {img_path}")
			return None
		image = cv2.imread(img_path)
		if image is None:
			logger.warning(f"Failed to read image: {img_path}")
		return image

	@staticmethod
	def _crop_labels(lbl_path, image, camera_name, scene_name, frame_id, folder_output_cropped, object_index):
		img_h, img_w = image.shape[:2]
		with open(lbl_path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			detection = Detection.parse(line)
			if detection is None:
				logger.warning(f"Invalid label format in {lbl_path}: {line.strip()}")
				continue
			if detection.is_too_small(img_w, img_h):
				continue

			x_min, y_min, x_max, y_max = detection.pixel_box(img_w, img_h)
			crop = image[y_min:y_max, x_min:x_max]
			img_name = f"{object_index:012d}_{camera_name}_{scene_name}_{int(frame_id):07d}_{x_min}_{y_min}_{x_max}_{y_max}_{detection.score:.5f}_{detection.class_id}.jpg"
			object_index = object_index + 1
			cv2.imwrite(os.path.join(folder_output_cropped, img_name), crop)
		return object_index


class DetectionJsonBuilder:
	IMG_W = 1920
	IMG_H = 1080

	def __init__(self, catalog, image_result_test):
		self.catalog = catalog
		self.image_result_test = image_result_test

	def build_scene(self, scene_name):
		scene_folder     = os.path.join(self.image_result_test, scene_name)
		folder_input_lbl = os.path.join(scene_folder, "detection/labels_renamed/")
		list_lbl = sorted(glob.glob(os.path.join(folder_input_lbl, "*.txt")), key=label_index_sort)

		full = {}
		person = {}
		non_person = {}
		object_index = 0
		pbar = tqdm(total=len(list_lbl))
		for lbl_path in list_lbl:
			lbl_index    = int(os.path.splitext(os.path.basename(lbl_path))[0].split("_")[-1])
			frame_id     = lbl_index % 9000
			camera_name  = Camera.adjust_camera_id(str(lbl_index // 9000))
			camera_index = f"{lbl_index // 9000:04d}"

			pbar.set_description(f"Processing building json {scene_name} -- {camera_name} -- {frame_id}")

			if not os.path.exists(lbl_path):
				logger.warning(f"Label not found for label: {lbl_path}")
				logger.warning(f"Expected label path: {lbl_path}")
				continue

			object_index = self._accumulate(lbl_path, frame_id, camera_index, object_index, full, person, non_person)
			pbar.update(1)
		pbar.close()

		self._save(scene_folder, scene_name, full, person, non_person)

	def _accumulate(self, lbl_path, frame_id, camera_index, object_index, full, person, non_person):
		with open(lbl_path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			detection = Detection.parse(line)
			if detection is None:
				logger.warning(f"Invalid label format in {lbl_path}: {line.strip()}")
				continue
			if detection.is_too_small(self.IMG_W, self.IMG_H):
				continue

			x_min, y_min, x_max, y_max = detection.pixel_box(self.IMG_W, self.IMG_H)
			record = [object_index, x_min, y_min, x_max, y_max, detection.score, detection.class_id]
			self._append(full, frame_id, camera_index, record)
			if self.catalog.is_person(detection.class_id):
				self._append(person, frame_id, camera_index, record)
			else:
				self._append(non_person, frame_id, camera_index, record)
			object_index = object_index + 1
		return object_index

	@staticmethod
	def _append(store, frame_id, camera_index, record):
		frame_key = str(frame_id)
		if frame_key not in store:
			store[frame_key] = {}
		if camera_index not in store[frame_key]:
			store[frame_key][camera_index] = []
		store[frame_key][camera_index].append(record)

	@staticmethod
	def _save(scene_folder, scene_name, full, person, non_person):
		outputs = [
			(f"{scene_name}_json_cropped_test_full.json", full),
			(f"{scene_name}_json_cropped_test_person.json", person),
			(f"{scene_name}_json_cropped_test_non-person.json", non_person),
		]
		for filename, data in outputs:
			with open(os.path.join(scene_folder, filename), 'w') as f:
				json.dump(data, f)


def main():
	parser = argparse.ArgumentParser(description="Process crop detection image.")
	parser.add_argument('--image_result_test' , type = str, default = "/media/vsw/Data1/MTMC_Tracking_2025/ExtractFrames/image_result_test/", help = "Folder where result out")
	args = parser.parse_args()

	catalog      = ObjectTypeCatalog()
	renamer      = LabelRenamer(args.image_result_test)
	cropper      = BboxCropper(args.image_result_test)
	json_builder = DetectionJsonBuilder(catalog, args.image_result_test)

	list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	for scene_name in tqdm(list_scene):
		renamer.rename_scene(scene_name)
		cropper.crop_scene(scene_name)
		json_builder.build_scene(scene_name)


if __name__ == "__main__":
	main()
