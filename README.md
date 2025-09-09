# AIC25 Track 01

Track 1: Multi-Camera 3D Perception of Automation Lab at Sungkyunkwan University

Paper: "DepthTrack: Cluster Meets BEV for Multi-Camera Multi-Target 3D Tracking"

---
## I. Dataset preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2025-track1/

Download dataset to the folder **<MTMC_Tracking_2025>**

The dataset folder structure should be as following:

```shell
<MTMC_Tracking_2025>
│   ├── test
│   │   ├── Warehouse_017
│   │   │   ├── videos
│   │   │   ├── depth_maps
│   │   │   ├── calibration.json
│   │   │   └── map.png
│   │   ├── Warehouse_018
│   │   ├── Warehouse_019
│   │   └── Warehouse_020
...
```

##### b. Data enhance download:

Download the data enhance files from the link below and put them in the folder **<MTMC_Tracking_2025>/test/Warehouse_018/videos** to replace the original.

- https://drive.google.com/drive/folders/1ny4Co3uV9gL-kJngGJn-4HoGztgtFGvT?usp=drive_link

```shell
<MTMC_Tracking_2025>
│   ├── test
│   │   ├── Warehouse_017
│   │   ├── Warehouse_018
│   │   │   ├── videos
│   │   │   │   ├── Camera_0000.mp4
│   │   │   │   ├── ...
│   │   │   │   └── Camera_0008.mp4
```

---
## II. Environment setup

#### a. Installation Miniconda or Anaconda:

1. Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

#### b. Create conda environment:

Follow the instructions in the following files to install the required dependencies.

- setup/setup_aic25_track1.sh
- setup/setup_kpr.sh
- setup/setup_mmpose.sh
- setup/setup_tracking.sh

```shell
AIC25_Track_01
│   ├── setup
│   │   ├── setup_aic25_track1.sh
│   │   ├── setup_kpr.sh
│   │   ├── setup_mmpose.sh
│   │   └── setup_tracking.sh
```

#### c. Load weights:

Downnoad each weight file and move them into the corresponding folder:

- https://drive.google.com/drive/folders/1AsPDJ1CWYkqO5Njg1lBC8HwIWTI-1Hq5?usp=sharing

Move folder **aic25_track1** into:

- model_zoo/aic25_track1

```shell
AIC25_Track_01
│   ├── model_zoo
│   │   └── aic25_track1
```

Move weight **swin_base_patch4_window7_224_22k.pth** into:

- ultilities/pose_reid_preprocess/keypoint_promptable_reidentification/pretrained_models/SOLIDER/swin_base_patch4_window7_224_22k.pth

```shell
AIC25_Track_01
│   ├── ultilities
│   │   ├── pose_reid_preprocess
│   │   │   ├── keypoint_promptable_reidentification
│   │   │   │   ├── pretrained_models
│   │   │   │   │   ├── SOLIDER
│   │   │   │   │   │   ├── swin_base_patch4_window7_224_22k.pth
```

Move weight **td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth** into:

- ultilities/pose_reid_preprocess/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth

```shell
AIC25_Track_01
│   ├── ultilities
│   │   ├── pose_reid_preprocess
│   │   │   ├── td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth
```

---
## III. Inference

#### a. Adjust the configuration

Default link to the dataset in the code is **/media/vsw/Data1/MTMC_Tracking_2025/**. You need to change it to your own path.

Change the varible ***ROOT_DATA_FOLDER*** in the file **ultilities/pose_reid_preprocess/configuration.py** to the folder path of the dataset **<MTMC_Tracking_2025>**.

- ROOT_DATA_FOLDER = <MTMC_Tracking_2025>  (example ROOT_DATA_FOLDER ='/media/vsw/Data1/MTMC_Tracking_2025/')

Change the varible ***ROOT_DATA_FOLDER*** in the file **ultilities/tracking/configuration.py** to the folder path of the dataset **<MTMC_Tracking_2025>**.

- ROOT_DATA_FOLDER = <MTMC_Tracking_2025>  (example ROOT_DATA_FOLDER ='/media/vsw/Data1/MTMC_Tracking_2025/')

Change the varible ***FOLDER_ROOT*** in the file **ultilities/configuration.py** to the folder path of the folder contain dataset **<MTMC_Tracking_2025>**.

- FOLDER_ROOT         = dirname(<MTMC_Tracking_2025>)  (example FOLDER_ROOT = '/media/vsw/Data1/')
- FOLDER_DATA_VERSION = "MTMC_Tracking_2025"

#### b. Run the code

Run the following commands in the terminal:

```shell
# Run all commands below from the folder AIC25_TRack_01

conda activate aic25_track1

bash run_inference_series.sh

cd ultilities/pose_reid_preprocess
conda activate mmpose
python run_pose.py 
cd ../..

cd ultilities/pose_reid_preprocess/keypoint_promptable_reidentification
conda activate kpr
python run_kpr_aicity25.py
cd ../../..

cd ultilities/tracking
conda activate tracking
python sv_tracking.py
python mv_tracking_prepare.py
python mv_tracking.py
cd ../..

conda activate aic25_track1
python ultilities/mapping_3d.py
python ultilities/filter_objects_out_bev.py
    
```

After running all command above, the output files will be in the folder 

- **<MTMC_Tracking_2025>/ExtractFrames/lookup_table/final_result_filtered.txt**

```shell
<MTMC_Tracking_2025>
│   ├── ExtractFrames
│   │   ├── lookup_table
│   │   │   └── final_result_filtered.txt
```

---
## IV. Training (optional)

Training dataset:

- https://drive.google.com/drive/folders/1ZETSfAbYNwVBoc_GBjWz3mjYhp2HgNRt?usp=sharing

---
## V. Acknowledgement

Most of the code is adapted from [Mon](https://github.com/phlong3105/mon).

This repository also features code from
[Ultralytics](https://github.com/ultralytics/ultralytics),
[Torchreid](https://github.com/KaiyangZhou/deep-person-reid),
[MMPose](https://github.com/open-mmlab/mmpose)
and [Bot-Sort](https://github.com/NirAharon/BoT-SORT)