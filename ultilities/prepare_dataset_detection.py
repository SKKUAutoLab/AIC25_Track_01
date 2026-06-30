import argparse
import glob
import os
import shutil

from loguru import logger
from tqdm import tqdm


def custom_file_sort(file_path):
	basename_noext = os.path.splitext(os.path.basename(file_path))[0]
	return int(basename_noext.split("_")[-1])


class VideoMerger:
	def __init__(self, folder_input):
		self.folder_input = folder_input

	def merge(self, scene_name):
		scene_folder  = os.path.join(self.folder_input, scene_name)
		videos_folder = os.path.join(scene_folder, "videos/")
		list_files = sorted(glob.glob(os.path.join(videos_folder, "*.mp4")))

		if len(list_files) == 0:
			logger.warning(f"No video files found in {videos_folder}.")
			return

		self._log_inputs(list_files)
		concat_file = self._write_concat_file(scene_folder, list_files)
		self._run_concat(scene_folder, scene_name, len(list_files), concat_file)

	@staticmethod
	def _log_inputs(list_files):
		list_files_str = ""
		for file_path in list_files:
			list_files_str += f"{os.path.basename(file_path)}, "
		logger.info(f"List files to merge: [{list_files_str}]")

	@staticmethod
	def _write_concat_file(scene_folder, list_files):
		concat_file = os.path.join(scene_folder, "file_list.txt")
		with open(concat_file, 'w') as f:
			for video_file in list_files:
				f.write(f"file 'videos/{os.path.basename(video_file)}'\n")
		return concat_file

	@staticmethod
	def _run_concat(scene_folder, scene_name, video_count, concat_file):
		output_file = os.path.join(scene_folder, f"{scene_name}.mp4")
		try:
			current_dir = os.getcwd()
			os.chdir(scene_folder)

			cmd = f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy -y {scene_name}.mp4"

			logger.info(f"Merging {video_count} videos into {output_file}")
			result = os.system(cmd)

			if result == 0:
				logger.success(f"Successfully merged videos to {output_file}")
			else:
				logger.error(f"FFmpeg command failed with exit code: {result}")

		except Exception as e:
			logger.error(f"Error merging videos: {e}")
		finally:
			os.chdir(current_dir)
			if os.path.exists(concat_file):
				os.remove(concat_file)


class FrameExtractor:
	def __init__(self, folder_input, folder_output):
		self.folder_input = folder_input
		self.folder_output = folder_output

	def extract(self, scene_name):
		scene_folder  = os.path.join(self.folder_input, scene_name)
		videos_folder = os.path.join(scene_folder, "videos/")
		list_files = sorted(glob.glob(os.path.join(videos_folder, "*.mp4")), key=custom_file_sort)

		if len(list_files) == 0:
			logger.warning(f"No video files found in {videos_folder}.")
			return

		output_scene = os.path.join(self.folder_output, scene_name)
		os.makedirs(output_scene, exist_ok=True)
		self._copy_map(scene_folder, scene_name)

		for video_file in tqdm(list_files, desc=f"Extracting images from videos in {scene_name}"):
			self._extract_video(video_file, output_scene)

	def _copy_map(self, scene_folder, scene_name):
		map_file = os.path.join(scene_folder, "map.png")
		if not os.path.exists(map_file):
			return
		destination = os.path.join(self.folder_output, scene_name, "map.png")
		shutil.copy(map_file, destination)
		logger.info(f"Copied map file to {destination}")

	@staticmethod
	def _extract_video(video_file, output_scene):
		video_name = os.path.splitext(os.path.basename(video_file))[0]
		output_video = os.path.join(output_scene, video_name)
		os.makedirs(output_video, exist_ok=True)

		cmd = f"ffmpeg -i {video_file} -start_number 0 {output_video}/%08d.jpg"
		logger.info(f"Extracting images from {video_file} to {output_video}")
		result = os.system(cmd)

		if result != 0:
			logger.error(f"FFmpeg command failed for {video_file} with exit code: {result}")


def main():
	parser = argparse.ArgumentParser(description="Process prepare detection dataset.")
	parser.add_argument('--input_dataset' , type = str, default = "/media/vsw/Data1/MTMC_Tracking_2025/test/", help = "Folder where dataset")
	parser.add_argument('--output_dataset' , type = str, default = "/media/vsw/Data1/MTMC_Tracking_2025/ExtractFrames/images_extract_full/", help = "Folder where result out")
	args = parser.parse_args()

	folder_intput = args.input_dataset
	folder_output = args.output_dataset
	list_scene    = ["Warehouse_017", "Warehouse_018", "Warehouse_019", "Warehouse_020"]

	merger    = VideoMerger(folder_intput)
	extractor = FrameExtractor(folder_intput, folder_output)

	pbar = tqdm(list_scene, desc="Adjust camera IDs in calibration files")
	for scene_name in tqdm(list_scene):
		pbar.set_description(f"Preparing detection dataset scene: {scene_name}")
		merger.merge(scene_name)
		extractor.extract(scene_name)
		pbar.update(1)
	pbar.close()


if __name__ == "__main__":
	main()
