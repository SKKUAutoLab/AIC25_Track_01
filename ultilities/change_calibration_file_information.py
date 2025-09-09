import argparse
import glob
import json
import os
import shutil
from copy import deepcopy

from loguru import logger
from tqdm import tqdm

from mtmc.core.objects.units import Camera

def rename_file_videos_depthmaps(folder_intput, scene_name):
	# Rename video
	folder_input_videos  = os.path.join(folder_intput, scene_name, "videos/")
	list_files = glob.glob(os.path.join(folder_input_videos, "*.mp4"))

	for file_path in tqdm(list_files, desc=f"Rename videos in scene name {scene_name} : "):
		file_path_old     = file_path
		file_name_old     = os.path.basename(file_path)
		file_name_old_ext = os.path.splitext(file_name_old)[0]

		file_name_new       = f'{Camera.adjust_camera_id(file_name_old_ext)}.mp4'
		file_path_new       = os.path.join(folder_input_videos, file_name_new)

		try:
			# Rename the folder
			os.rename(file_path_old, file_path_new)
			logger.info(f"Folder '{file_path_old}' successfully renamed to '{file_path_new}'.")
		except FileNotFoundError:
			logger.error(f"Error: Folder '{file_path_old}' not found.")
		except FileExistsError:
			logger.error(f"Error: Folder '{file_path_new}' already exists.")
		except Exception as e:
			logger.error(f"An unexpected error occurred: {e}")

	# Rename depthmaps
	folder_input_depthmaps  = os.path.join(folder_intput, scene_name, "depth_maps/")
	list_files = glob.glob(os.path.join(folder_input_depthmaps, "*.h5"))

	for file_path in tqdm(list_files, desc=f"Rename depth_maps in scene name {scene_name} : "):
		file_path_old     = file_path
		file_name_old     = os.path.basename(file_path)
		file_name_old_ext = os.path.splitext(file_name_old)[0]

		file_name_new       = f'{Camera.adjust_camera_id(file_name_old_ext)}.h5'
		file_path_new       = os.path.join(folder_input_depthmaps, file_name_new)

		try:
			# Rename the folder
			os.rename(file_path_old, file_path_new)
			logger.info(f"Folder '{file_path_old}' successfully renamed to '{file_path_new}'.")
		except FileNotFoundError:
			logger.error(f"Error: Folder '{file_path_old}' not found.")
		except FileExistsError:
			logger.error(f"Error: Folder '{file_path_new}' already exists.")
		except Exception as e:
			logger.error(f"An unexpected error occurred: {e}")


def adjust_camera_id_calibration(json_file):

	# Load the file
	with open(json_file, "r") as f_read:
		data = json.load(f_read)

	# Replace id for each sensor of type "camera"
	for sensor in data.get("sensors", []):
		if sensor.get("type") == "camera":
			sensor["id"] = Camera.adjust_camera_id(sensor["id"])

	# Optionally, save to a new file
	json_file = json_file.replace("calibration.json", "calibration_modified.json")
	with open(json_file, "w") as f_write:
		json.dump(data, f_write, indent=4)

	print(f"All camera IDs have been replaced and saved to {json_file}.")

def main():
	parser = argparse.ArgumentParser(description="Process crop detetion image.")
	parser.add_argument('--input_test' , type = str, default = "/media/vsw/Data1/MTMC_Tracking_2025/test/", help = "Folder where result out")
	args = parser.parse_args()

	folder_intput = args.input_test
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	# NOTE: Adjust camera IDs in each JSON file
	for scene_name in tqdm(list_scene):
		adjust_camera_id_calibration(os.path.join(folder_intput, scene_name, f"calibration.json"))
		rename_file_videos_depthmaps(folder_intput, scene_name)

if __name__ == "__main__":
	main()
