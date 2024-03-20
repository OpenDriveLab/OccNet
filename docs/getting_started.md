## Installation

Follow https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md to prepare the environment.

## Preparing Dataset

1. Download the nuScenes dataset and put in into `data/nuscenes`

2. Download our `openocc_v2.1.zip` and `infos.zip` from [HERE](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq) and unzip them in `data/nuscenes`. 

3. Organize your folder structure as below：

```
data/nuscenes
├── maps
├── nuscenes_infos_train_occ.pkl
├── nuscenes_infos_val_occ.pkl
├── openocc_v2
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```

## Training

```
./tools/dist_train.sh projects/configs/bevformer/bevformer_base_occ.py 8
```

## Testing

```
export CUDA_VISIBLE_DEVICES=0
./tools/dist_test.sh projects/configs/bevformer/bevformer_base_occ.py work_dirs/bevformer_base_occ/epoch_24.pth 1
```

## Test Submission

1. Fill your information in `projects/mmdet3d_plugin/datasets/nuscenes_occ.py` (Line 231).

2. Test the baseline model on the test split with 8 GPUs, and generate the submission to the official evaluation server.

```
./tools/dist_test.sh projects/configs/bevformer/bevformer_base_occ_test.py work_dirs/bevformer_base_occ/epoch_24.pth 8 --format-only --eval-options 'submission_prefix=./occ_submission'
```

3. The submission file is located in `occ_submission/submission.gz`. Good luck!

### Performance

TBD
