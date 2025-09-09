#!/bin/bash

# stop at the first error
set -e

export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # .
export DIR_TSS=$DIR_CURRENT                         # .

FOLDER_DATASET="/media/vsw/Data1/MTMC_Tracking_2025"
PATH_CONDA_SH="/home/vsw/miniconda3/etc/profile.d/conda.sh"

START_TIME="$(date -u +%s.%N)"
###########################################################################################################
echo "###########################"
echo "STARTING"
echo "###########################"

# NOTE: ACTIVATE ENVIRONMENT
echo "********************"
echo "ACTIVATE ENVIRONMENT"
echo "********************"
source $PATH_CONDA_SH
conda activate aic25_track1
echo "Active Conda environment: $CONDA_DEFAULT_ENV"


echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

# NOTE: EXTRACT IMAGES PROCESS
echo "**********************"
echo "EDIT INFORMATION"
echo "**********************"
python ultilities/change_calibration_file_information.py  \
 --input_test ${FOLDER_DATASET}/test/

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

# NOTE: EXTRACT IMAGES PROCESS
echo "*************************"
echo "PREPARE DATASET DETECTION"
echo "*************************"
#mkdir -p ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_017/detection/images/
#mkdir -p ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_018/detection/images/
#mkdir -p ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_019/detection/images/
#mkdir -p ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_020/detection/images/
#
#ffmpeg -i ${FOLDER_DATASET}/test/Warehouse_017/Warehouse_017.mp4 -start_number 0 ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_017/detection/images/%08d.jpg &
#ffmpeg -i ${FOLDER_DATASET}/test/Warehouse_018/Warehouse_018.mp4 -start_number 0 ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_018/detection/images/%08d.jpg &
#ffmpeg -i ${FOLDER_DATASET}/test/Warehouse_018/Warehouse_018.mp4 -start_number 0 ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_018/detection/images/%08d.jpg &
#ffmpeg -i ${FOLDER_DATASET}/test/Warehouse_019/Warehouse_019.mp4 -start_number 0 ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_019/detection/images/%08d.jpg &
#ffmpeg -i ${FOLDER_DATASET}/test/Warehouse_020/Warehouse_020.mp4 -start_number 0 ${FOLDER_DATASET}/ExtractFrames/image_result_test/Warehouse_020/detection/images/%08d.jpg
#wait

python ultilities/prepare_dataset_detection.py  \
  --input_dataset  ${FOLDER_DATASET}/test/  \
  --output_dataset ${FOLDER_DATASET}/ExtractFrames/images_extract_full/

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

# NOTE: DETECTION PROCESS
echo "*****************"
echo "DETECTION PROCESS"
echo "*****************"
# Define an array
LIST_VIDEO=(
  "${FOLDER_DATASET}/test/Warehouse_017/Warehouse_017.mp4"
  "${FOLDER_DATASET}/test/Warehouse_018/Warehouse_018.mp4"
  "${FOLDER_DATASET}/test/Warehouse_019/Warehouse_019.mp4"
  "${FOLDER_DATASET}/test/Warehouse_020/Warehouse_020.mp4"
)

LIST_MODEL=(
  "${DIR_CURRENT}/model_zoo/aic25_track1/aic25_track1_yolo11x_imz_1920_warehouse_017/weights/best.pt"
  "${DIR_CURRENT}/model_zoo/aic25_track1/aic25_track1_yolo11x_imz_1920_warehouse_018/weights/best.pt"
  "${DIR_CURRENT}/model_zoo/aic25_track1/aic25_track1_yolo11x_imz_1920_warehouse_019/weights/best.pt"
  "${DIR_CURRENT}/model_zoo/aic25_track1/aic25_track1_yolo11x_imz_1920_warehouse_020/weights/best.pt"
)

FOLDER_OUTPUT="${FOLDER_DATASET}/ExtractFrames/image_result_test/"
IMGSZ=1920
DEVICE=0
CONF=0.25

for index in "${!LIST_VIDEO[@]}"; do
filename_no_ext=$(basename ${LIST_VIDEO[$index]} .mp4)
echo "Index: $index, Value: ${LIST_VIDEO[$index]}"
echo "File name no extension ${filename_no_ext}"
echo "Weight ${LIST_MODEL[$index]}"
echo ""

# Detection
OUTPUT="${FOLDER_OUTPUT}${filename_no_ext}/detection/"
yolo predict \
  model=${LIST_MODEL[$index]} \
  source=${LIST_VIDEO[$index]} \
  imgsz=$IMGSZ \
  device=$index \
  conf=$CONF \
  batch=2  \
  project=$(dirname "$OUTPUT") \
  name=$(basename "$OUTPUT") \
  save_txt=True  \
  save_conf=True  \
  save=True  \
  verbose=False &
done
wait

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

# NOTE: POST DETECTION PROCESS
echo "**********************"
echo "POST DETECTION PROCESS"
echo "**********************"
python ultilities/crop_bbox_yolo.py  \
  --image_result_test ${FOLDER_DATASET}/ExtractFrames/image_result_test/

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

echo "###########################"
echo "ENDING"
echo "###########################"
###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."