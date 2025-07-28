import os
import sys
import glob
import json

import multiprocessing
import threading

import cv2
from tqdm import tqdm
from loguru import logger

from mtmc.core.objects.units import Camera

object_type_name = {
	0 : "Person", # red
	1 : "Forklift", # green
	2 : "NovaCarter", # blue
	3 : "Transporter", # yellow
	4 : "FourierGR1T2", # purple
	5 : "AgilityDigit", # pink
}

object_type_id = {
	"Person"       : 0, # red
	"Forklift"     : 1, # green
	"NovaCarter"   : 2, # blue
	"Transporter"  : 3, # yellow
	"FourierGR1T2" : 4, # purple
	"AgilityDigit" : 5, # pink
}


def crop_image_from_all_cam_each_scene(scene_name):
	def custom_file_sort(file_path):
		basename       = os.path.basename(file_path)
		basename_noext = os.path.splitext(basename)[0]
		file_index     = basename_noext.split("_")[-1]
		return int(file_index)

	# Initialize paths
	folder_input = "/media/vsw/SSD_2/1_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_result/"
	folder_input_img      = os.path.join(folder_input, scene_name, "cycle_2_dect/images")
	folder_input_lbl      = os.path.join(folder_input, scene_name, "cycle_2_dect/labels")
	folder_output_cropped = os.path.join(folder_input, "image_croped_test", scene_name)
	os.makedirs(folder_output_cropped, exist_ok=True)

	list_lbl = sorted(glob.glob(os.path.join(folder_input_lbl, "*.txt")), key=custom_file_sort)

	object_index = 0
	camera_name  = "Camera_0000"
	pbar = tqdm(total=len(list_lbl))
	for label_index, lbl_path in enumerate(list_lbl):

		image_index  = int(os.path.splitext(os.path.basename(lbl_path))[0])
		frame_id     = image_index % 9000
		camera_name  = Camera.adjust_camera_id(str(image_index // 9000))

		pbar.set_description(f"Processing {scene_name} -- {camera_name} -- {frame_id}")

		# if label_index % (len(list_lbl) // 300) != 0:
		# 	pbar.update(1)
		# 	continue

		img_path = os.path.join(folder_input_img, os.path.basename(lbl_path).replace(".txt", ".jpg"))

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

def create_json_from_cropped_images(scene_name):
	def custom_file_sort(file_path):
		basename       = os.path.basename(file_path)
		basename_noext = os.path.splitext(basename)[0]
		file_index     = basename_noext.split("_")[-1]
		return int(file_index)

	# Initialize paths
	folder_input = "/media/vsw/SSD_2/1_Dataset/AI_City_Challenge/2025/Track_1/MTMC_Tracking_2025/ExtractFrames/image_result/"
	folder_input_img           = os.path.join(folder_input, scene_name, "cycle_2_dect/images")
	folder_input_lbl           = os.path.join(folder_input, scene_name, "cycle_2_dect/labels")
	folder_output_cropped_json = os.path.join(folder_input, "json_cropped_test", "forklift", f"{scene_name}.json")
	os.makedirs(os.path.dirname(folder_output_cropped_json), exist_ok=True)

	list_img = sorted(glob.glob(os.path.join(folder_input_img, "*.jpg")), key=custom_file_sort)

	object_index = 0
	camera_name  = "Camera_0000"
	pbar = tqdm(total=len(list_img))
	json_data = {}
	for img_index, img_path in enumerate(list_img):

		image_index  = int(os.path.splitext(os.path.basename(img_path))[0])
		frame_id     = image_index % 9000

		camera_name  = Camera.adjust_camera_id(str(image_index // 9000))
		camera_index = f"{image_index // 9000:04d}"

		pbar.set_description(f"Processing {scene_name} -- {camera_name} -- {frame_id}")

		lbl_path = os.path.join(folder_input_lbl, os.path.basename(img_path).replace(".jpg", ".txt"))

		if not os.path.exists(lbl_path):
			logger.warning(f"Label not found for label: {lbl_path}")
			logger.warning(f"Expected label path: {lbl_path}")
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
				width    = float(components[3])
				height   = float(components[4])
				score    = float(components[5])

				# Convert YOLO format to bounding box coordinates
				x_min = int((x_center - width / 2) * img_w)
				y_min = int((y_center - height / 2) * img_h)
				x_max = int((x_center + width / 2) * img_w)
				y_max = int((y_center + height / 2) * img_h)

				if object_type_name[class_id] not in ["Forklift"]:
					continue

				if width * img_w < 10 or height * img_h < 10:
					continue

				# Crop the bounding box
				# img_name = f"{object_index:012d}_{camera_name}_{scene_name}_{int(frame_id):07d}_{x_min}_{y_min}_{x_max}_{y_max}_{score:.5f}_{class_id}.jpg"


				if str(frame_id) not in json_data:
					json_data[str(frame_id)] = {}
				if camera_index not in json_data[str(frame_id)]:
					json_data[str(frame_id)][str(camera_index)] = []
				json_data[str(frame_id)][str(camera_index)].append([
					object_index,
					x_min,
					y_min,
					x_max,
					y_max,
					score,
					class_id
				])
				object_index = object_index + 1

		# DEBUG:
		# if img_index > 200:
		# 	break

		pbar.update(1)
	pbar.close()

	# write json data to file
	with open(folder_output_cropped_json, 'w') as f:
		json.dump(json_data, f)


def run_multi_thread():
	list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_018-enhance", "Warehouse_019", "Warehouse_020"]

	threads = []
	for scene_name in list_scene:
		# t = threading.Thread(target=crop_image_from_all_cam_each_scene, args=(scene_name,))
		t = threading.Thread(target=create_json_from_cropped_images, args=(scene_name,))
		threads.append(t)

	# starting threads
	for i in range(len(list_scene)):
		threads[i].start()

	# wait until threads are finished
	for i in range(len(list_scene)):
		threads[i].join()

def run_single_thread():
	# list_scene = ["Warehouse_017", "Warehouse_018", "Warehouse_018-enhance", "Warehouse_019", "Warehouse_020"]
	list_scene = ["Warehouse_020"]
	for scene_name in tqdm(list_scene):
		# crop_image_from_all_cam_each_scene(scene_name)
		create_json_from_cropped_images(scene_name)

def main():
	# run_single_thread()

	run_multi_thread()

	pass


if __name__ == "__main__":
	main()

