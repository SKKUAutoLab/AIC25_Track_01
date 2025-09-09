import argparse
import os
import sys
import glob
import json
import shutil

import multiprocessing
import threading

import cv2
from tqdm import tqdm
from loguru import logger

from mtmc.core.objects.units import Camera
from ultilities import configuration

object_type_name = {
	0: "Person"      , # red
	1: "Forklift"    , # green
	2: "NovaCarter"  , # blue
	3: "Transporter" , # yellow
	4: "FourierGR1T2", # purple
	5: "AgilityDigit", # pink
}

object_type_id = {
	"Person"       : 0, # red
	"Forklift"     : 1, # green
	"NovaCarter"   : 2, # blue
	"Transporter"  : 3, # yellow
	"FourierGR1T2" : 4, # purple
	"AgilityDigit" : 5, # pink
}

def is_number(s):
	try:
		float(s)  # or use int(s) for integers only
		return True
	except ValueError:
		return False

def rename_files(scene_name, image_result_test):
	# init folder
	# folder_input  = "/media/vsw/Data1/MTMC_Tracking_2025/ExtractFrames/image_result_test/"
	folder_input  = os.path.join(image_result_test, scene_name, "detection/labels/")
	folder_output = os.path.join(image_result_test, scene_name, "detection/labels_renamed/")
	list_txt      = glob.glob(os.path.join(folder_input, "*.txt"))

	os.makedirs(folder_output, exist_ok=True)

	for txt_path in tqdm(list_txt, desc=f"Rename file in scene name {scene_name} : "):
		txt_path_old     = txt_path
		txt_name_old     = os.path.basename(txt_path)
		txt_name_old_ext = os.path.splitext(txt_name_old)[0]

		txt_name_new_index = int(txt_name_old.split("_")[-1].split(".")[0]) - 1
		txt_name_new       = f'{txt_name_new_index:08d}.txt'
		txt_path_new       = os.path.join(folder_output, txt_name_new)

		# DEBUG:
		# print(f"{txt_name_old} -- {txt_name_new}")

		try:
			shutil.copy(txt_path_old, txt_path_new)
			# print(f"'{txt_path_old}' copied to '{txt_path_new}' (overwritten if existed).")
		except FileNotFoundError:
			# print(f"Error: Source file '{txt_path_old}' not found.")
			pass
		except OSError as e:
			# print(f"Error copying file: {e}")
			pass

def crop_image_from_all_cam_each_scene(scene_name, image_result_test):
	def custom_file_sort(file_path):
		basename       = os.path.basename(file_path)
		basename_noext = os.path.splitext(basename)[0]
		file_index     = basename_noext.split("_")[-1]
		return int(file_index)

	# Initialize paths
	# folder_input = "/media/vsw/SSD_2/1_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_result/"
	folder_input          = image_result_test
	folder_input_img      = os.path.join(folder_input, scene_name, "detection/images")
	folder_input_lbl      = os.path.join(folder_input, scene_name, "detection/labels_renamed")
	folder_output_cropped = os.path.join(folder_input, scene_name, "image_croped_test")
	os.makedirs(folder_output_cropped, exist_ok=True)

	list_lbl = sorted(glob.glob(os.path.join(folder_input_lbl, "*.txt")), key=custom_file_sort)

	# DEBUG:
	# print(os.path.join(folder_input_lbl, "*.txt"))

	object_index = 0
	camera_name  = "Camera_0000"
	pbar = tqdm(total=len(list_lbl))
	for label_index, lbl_path in enumerate(list_lbl):

		image_index  = int(os.path.splitext(os.path.basename(lbl_path))[0])
		frame_id     = image_index % 9000
		camera_name  = Camera.adjust_camera_id(str(image_index // 9000))

		# DEBUG:
		# if frame_id >= 500:
		# 	continue

		pbar.set_description(f"Processing croping {scene_name} -- {camera_name} -- {frame_id}")

		# if label_index % (len(list_lbl) // 300) != 0:
		# 	pbar.update(1)
		# 	continue

		# NOTE: Construct the image path from the label path
		# img_path = os.path.join(folder_input_img, os.path.basename(lbl_path).replace(".txt", ".jpg"))
		img_path = os.path.join(configuration.FOLDER_INPUT_FULL_EXTRACTION_IMAGE, scene_name, camera_name, f"{frame_id:08d}.jpg")

		if not os.path.exists(img_path):
			logger.warning(f"Image not found for label: {lbl_path}")
			logger.warning(f"Expected image path: {img_path}")
			continue

		# Read the image
		image = cv2.imread(img_path)
		if image is None:
			logger.warning(f"Failed to read image: {img_path}")
			continue

		img_h, img_w = image.shape[:2]

		# Read label file
		with open(lbl_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				# Split the line into components
				components = line.strip().split()
				if len(components) < 5:
					logger.warning(f"Invalid label format in {lbl_path}: {line.strip()}")
					continue

				# Extract the bounding box coordinates
				class_id = int(components[0])
				x_center = float(components[1])
				y_center = float(components[2])
				width = float(components[3])
				height = float(components[4])
				score  = float(components[5])

				# Convert YOLO format to bounding box coordinates
				x_min = int((x_center - width / 2) * img_w)
				y_min = int((y_center - height / 2) * img_h)
				x_max = int((x_center + width / 2) * img_w)
				y_max = int((y_center + height / 2) * img_h)

				# if object_type_name[class_id] not in ["Person"]:
				# 	continue

				if width * img_w < 10 or height * img_h < 10:
					continue

				# Crop the bounding box
				crop = image[y_min:y_max, x_min:x_max]
				img_name = f"{object_index:012d}_{camera_name}_{scene_name}_{int(frame_id):07d}_{x_min}_{y_min}_{x_max}_{y_max}_{score:.5f}_{class_id}.jpg"
				object_index = object_index + 1
				output_path = os.path.join(folder_output_cropped, img_name)
				cv2.imwrite(output_path, crop)

		pbar.update(1)
	pbar.close()


def create_json_from_cropped_images(scene_name, image_result_test):
	def custom_file_sort(file_path):
		basename       = os.path.basename(file_path)
		basename_noext = os.path.splitext(basename)[0]
		file_index     = basename_noext.split("_")[-1]
		return int(file_index)

	# Initialize paths
	folder_input               = image_result_test
	folder_input_lbl           = os.path.join(folder_input, scene_name, "detection/labels_renamed/")
	folder_output_cropped_json_full       = os.path.join(folder_input, scene_name, f"{scene_name}_json_cropped_test_full.json")
	folder_output_cropped_json_person     = os.path.join(folder_input, scene_name, f"{scene_name}_json_cropped_test_person.json")
	folder_output_cropped_json_non_person = os.path.join(folder_input, scene_name, f"{scene_name}_json_cropped_test_non-person.json")

	list_lbl = sorted(glob.glob(os.path.join(folder_input_lbl, "*.txt")), key=custom_file_sort)

	object_index         = 0
	pbar                 = tqdm(total = len(list_lbl))
	json_data_full       = {}
	json_data_person     = {}
	json_data_non_person = {}
	img_w                = 1920
	img_h                = 1080
	for _, lbl_path in enumerate(list_lbl):

		lbl_index  = int(os.path.splitext(os.path.basename(lbl_path))[0].split("_")[-1])
		lbl_index  = lbl_index
		frame_id   = lbl_index % 9000

		camera_name  = Camera.adjust_camera_id(str(lbl_index // 9000))
		camera_index = f"{lbl_index // 9000:04d}"

		# DEBUG:
		# if frame_id >= 500:
		# 	continue

		pbar.set_description(f"Processing building json {scene_name} -- {camera_name} -- {frame_id}")

		if not os.path.exists(lbl_path):
			logger.warning(f"Label not found for label: {lbl_path}")
			logger.warning(f"Expected label path: {lbl_path}")
			continue

		# Read label file
		with open(lbl_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				# Split the line into components
				components = line.strip().split()
				if len(components) < 5:
					logger.warning(f"Invalid label format in {lbl_path}: {line.strip()}")
					continue

				# Extract the bounding box coordinates
				class_id = int(components[0])
				x_center = float(components[1])
				y_center = float(components[2])
				width    = float(components[3])
				height   = float(components[4])
				score    = float(components[5])

				# Convert YOLO format to bounding box coordinates
				x_min = int((x_center - width / 2) * img_w)
				y_min = int((y_center - height / 2) * img_h)
				x_max = int((x_center + width / 2) * img_w)
				y_max = int((y_center + height / 2) * img_h)

				if width * img_w < 10 or height * img_h < 10:
					continue

				# Crop the bounding box
				# img_name = f"{object_index:012d}_{camera_name}_{scene_name}_{int(frame_id):07d}_{x_min}_{y_min}_{x_max}_{y_max}_{score:.5f}_{class_id}.jpg"

				if str(frame_id) not in json_data_full:
					json_data_full[str(frame_id)] = {}
				if camera_index not in json_data_full[str(frame_id)]:
					json_data_full[str(frame_id)][str(camera_index)] = []
				# Full
				json_data_full[str(frame_id)][str(camera_index)].append([
					object_index,
					x_min,
					y_min,
					x_max,
					y_max,
					score,
					class_id
				])
				if object_type_name[class_id] in ["Person"]:
					if str(frame_id) not in json_data_person:
						json_data_person[str(frame_id)] = {}
					if camera_index not in json_data_person[str(frame_id)]:
						json_data_person[str(frame_id)][str(camera_index)] = []
					# person
					json_data_person[str(frame_id)][str(camera_index)].append([
						object_index,
						x_min,
						y_min,
						x_max,
						y_max,
						score,
						class_id
					])
				else:
					if str(frame_id) not in json_data_non_person:
						json_data_non_person[str(frame_id)] = {}
					if camera_index not in json_data_non_person[str(frame_id)]:
						json_data_non_person[str(frame_id)][str(camera_index)] = []
					# non person
					json_data_non_person[str(frame_id)][str(camera_index)].append([
						object_index,
						x_min,
						y_min,
						x_max,
						y_max,
						score,
						class_id
					])
				object_index = object_index + 1

		pbar.update(1)
	pbar.close()

	# write json data to file
	with open(folder_output_cropped_json_full, 'w') as f:
		json.dump(json_data_full, f)
	with open(folder_output_cropped_json_person, 'w') as f:
		json.dump(json_data_person, f)
	with open(folder_output_cropped_json_non_person, 'w') as f:
		json.dump(json_data_non_person, f)


def main():
	parser = argparse.ArgumentParser(description="Process crop detection image.")
	parser.add_argument('--image_result_test' , type = str, default = "/media/vsw/Data1/MTMC_Tracking_2025/ExtractFrames/image_result_test/", help = "Folder where result out")
	args = parser.parse_args()

	list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]
	# list_scene = ["Warehouse_018", "Warehouse_019", "Warehouse_020"]
	for scene_name in tqdm(list_scene):
		rename_files(scene_name, args.image_result_test)
		crop_image_from_all_cam_each_scene(scene_name, args.image_result_test)
		create_json_from_cropped_images(scene_name, args.image_result_test)
		pass

if __name__ == "__main__":
	main()

