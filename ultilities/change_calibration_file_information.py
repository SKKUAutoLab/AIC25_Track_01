import argparse
import glob
import json
import os

from loguru import logger
from tqdm import tqdm

from mtmc.core.objects.units import Camera


class CameraIdRenamer:
	def __init__(self, scene_root, scene_name):
		self.scene_root = scene_root
		self.scene_name = scene_name

	def rename_media(self):
		self._rename_folder("videos", "mp4")
		self._rename_folder("depth_maps", "h5")

	def _rename_folder(self, subfolder, extension):
		folder = os.path.join(self.scene_root, subfolder, "")
		list_files = glob.glob(os.path.join(folder, f"*.{extension}"))
		desc = f"Rename {subfolder} in scene name {self.scene_name} : "
		for file_path in tqdm(list_files, desc=desc):
			self._rename_one(folder, file_path, extension)

	@staticmethod
	def _rename_one(folder, file_path_old, extension):
		stem = os.path.splitext(os.path.basename(file_path_old))[0]
		file_path_new = os.path.join(folder, f"{Camera.adjust_camera_id(stem)}.{extension}")
		try:
			os.rename(file_path_old, file_path_new)
			logger.info(f"Folder '{file_path_old}' successfully renamed to '{file_path_new}'.")
		except FileNotFoundError:
			logger.error(f"Error: Folder '{file_path_old}' not found.")
		except FileExistsError:
			logger.error(f"Error: Folder '{file_path_new}' already exists.")
		except Exception as e:
			logger.error(f"An unexpected error occurred: {e}")


class CalibrationCameraIds:
	def __init__(self, json_file):
		self.json_file = json_file

	def adjust(self):
		data = self._load()
		self._replace_camera_ids(data)
		output_file = self.json_file.replace("calibration.json", "calibration_modified.json")
		self._save(data, output_file)
		print(f"All camera IDs have been replaced and saved to {output_file}.")

	def _load(self):
		with open(self.json_file, "r") as f_read:
			return json.load(f_read)

	@staticmethod
	def _replace_camera_ids(data):
		for sensor in data.get("sensors", []):
			if sensor.get("type") != "camera":
				continue
			sensor["id"] = Camera.adjust_camera_id(sensor["id"])

	@staticmethod
	def _save(data, output_file):
		with open(output_file, "w") as f_write:
			json.dump(data, f_write, indent=4)


def main():
	parser = argparse.ArgumentParser(description="Process crop detetion image.")
	parser.add_argument('--input_test' , type = str, default = "/media/vsw/Data1/MTMC_Tracking_2025/test/", help = "Folder where result out")
	args = parser.parse_args()

	folder_intput = args.input_test
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	for scene_name in tqdm(list_scene):
		scene_root = os.path.join(folder_intput, scene_name)
		CalibrationCameraIds(os.path.join(scene_root, "calibration.json")).adjust()
		CameraIdRenamer(scene_root, scene_name).rename_media()


if __name__ == "__main__":
	main()
