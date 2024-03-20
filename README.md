<div id="top" align="center">

# CVPR 2024 Autonomous Grand Challenge Occupancy and Flow

<a href="https://twitter.com/OpenDriveLab" target="_blank">
  <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/OpenDriveLab?style=social&color=brightgreen&logo=twitter" />
</a>
<a href="#license">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>


<img src="./figs/occupanc_1.gif" width="696px">

</div>

## Introduction

Understanding the 3D surroundings including the background stuffs and foreground objects is important for autonomous driving. In the traditional 3D object detection task, a foreground object is represented by the 3D bounding box. However, the geometrical shape of the object is complex, which can not be represented by a simple 3D box, and the perception of the background stuffs is absent. The goal of this task is to predict the 3D occupancy of the scene. In this task, we provide a large-scale occupancy benchmark based on the nuScenes dataset. The benchmark is a voxelized representation of the 3D space, and the occupancy state and semantics of the voxel in 3D space are jointly estimated in this task. The complexity of this task lies in the dense prediction of 3D space given the surround-view images.

If you use the challenge dataset in your paper, please consider citing OccNet with the following BibTex:

```bibtex
@article{sima2023_occnet,
      title={Scene as Occupancy},
      author={Chonghao Sima and Wenwen Tong and Tai Wang and Li Chen and Silei Wu and Hanming Deng and Yi Gu and Lewei Lu and Ping Luo and Dahua Lin and Hongyang Li},
      year={2023},
      eprint={2306.02851},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

- Official website: :globe_with_meridians: [AGC2024](https://huggingface.co/spaces/AGC2024-S/occupancy-and-flow-2024)
- Evaluation server: :hugs: [Hugging Face](https://huggingface.co/spaces/AGC2024-S/occupancy-and-flow-2024)

## News

- 2024-03-01: The challenge begins.
- 2024-03-14: We release a new version (`openocc_v2.1`) of the occupancy ground-truth, including some bug fixes regarding the occupancy flow. **Delete the old version and download the new one!** Please refer to [getting_started](docs/getting_started.md) for details.

## Table of Contents

- [CVPR 2024 Autonomous Grand Challenge Occupancy and Flow](#cvpr-2024-occupancy-and-flow-challenge)
  - [Introduction](#introduction)
  - [News](#news)
  - [Task Definition](#task-definition)
    - [Rules for Occupancy and Flow Challenge](#rules-for-occupancy-and-flow-challenge)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Ray-based mIoU](#evaluation-metrics)
    - [AVE for Occupancy Flow](#evaluation-metrics)
    - [Occupancy Score](#evaluation-metrics)
  - [Data](#data)
    - [Basic Information](#basic-information)
    - [Download](#download)
    - [Hierarchy](#hierarchy)
    - [Known Issues](#known-issues)
  - [Getting Started](#getting-started)
  - [Submission](#submission)
  - [License](#license)

## Task Definition

Given images from multiple cameras, the goal is to predict the semantics and flow of each voxel grid in the scene.

### Rules for Occupancy and Flow Challenge

* We allow using annotations provided in the nuScenes dataset. During inference, the input modality of the model should be camera only. 
* No future frame is allowed during inference.
* In order to check the compliance, we will ask the participants to provide technical reports to the challenge committee and the participant will be asked to provide a public talk about the method after winning the award.
* Every submission provides method information. We encourage publishing code, but do not make it a requirement.
* Each team can have at most one account on the evaluation server. Users that create multiple accounts to circumvent the rules will be excluded from the challenge.
* Each team can submit at most three results during the challenge. 
* Faulty submissions that return an error on HuggingFace do not count towards the submission limit.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from the challenge.

<p align="right">(<a href="#top">back to top</a>)</p>

## Evaluation Metrics

Leaderboard ranking for this challenge is by the **Occupancy Score**. It consists of two parts: **Ray-based mIoU**, and absolute velocity error for occupancy flow.

The implementation is here: [projects/mmdet3d_plugin/datasets/ray_metrics.py](https://github.com/OpenDriveLab/OccNet/blob/challenge/projects/mmdet3d_plugin/datasets/ray_metrics.py)

### Ray-based mIoU

We use the well-known mean intersection-over-union (mIoU) metric. However, the elements of the set are now query rays, not voxels.

Specifically, we emulate LiDAR by projecting query rays into the predicted 3D occupancy volume. For each query ray, we compute the distance it travels before it intersects any surface. We then retrieve the corresponding class label and flow prediction.

We apply the same procedure to the ground-truth occupancy to obtain the groud-truth depth, class label and flow.

A query ray is classified as a **true positive** (TP) if the class labels coincide **and** the L1 error between the ground-truth depth and the predicted depth is less than either a certain threshold (e.g. 2m).

Let $C$ be he number of classes. 

$$
mIoU=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$ , $FP_c$ , and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.

We finally average over distance thresholds of {1, 2, 4} meters and compute the mean across classes.

For more details about this metric, we will release a technical report within a few days, please stay tuned.

### AVE for Occupancy Flow

Here we measure velocity errors for a set of true positives (TP). We use a threshold of 2m distance.

The absolute velocity error (AVE) is defined for 8 classes ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian') in m/s. 

### Occupancy Score

The final occupancy score is defined to be a weighted sum of mIoU and mAVE. Note that the velocity errors are converted to velocity scores as `max(1 - mAVE, 0.0)`. That is,

```
OccScore = mIoU * 0.9 + max(1 - mAVE, 0.0) * 0.1
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Data

### Basic Information
<div align="center">
  
| Type |  Info |
| :----: | :----: |
| train           | 28,130 |
| val             | 6,019 |
| test            | 6,008 |
| cameras         | 6 |
| voxel size      | 0.4m |
| range           | [-40m, -40m, -1m, 40m, 40m, 5.4m]|
| volume size     | [200, 200, 16]|
| #classes        | 0 - 16 |
  
</div>

- The dataset contains 17 classes. Voxel semantics for each sample frame is given as `[semantics]` in the labels.npz. Occupancy flow is given as `[flow]`  in the labels.npz.


### Download


1. Download the nuScenes dataset and put in into `data/nuscenes`

2. Download our `openocc_v2.1.zip` and `infos.zip` from [OpenDataLab](https://opendatalab.com/OpenDriveLab/CVPR24-Occ-Flow-Challenge/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq)

3. Unzip them in `data/nuscenes`


### Hierarchy

The hierarchy of folder `data/nuscenes` is described below:

```
nuscenes
├── maps
├── nuscenes_infos_train_occ.pkl
├── nuscenes_infos_val_occ.pkl
├── openocc_v2
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```

- `openocc_v2` is the occuapncy GT.
- `nuscenes_infos_train_occ.pkl` and `nuscenes_infos_val_occ.pkl` contains meta infos of the dataset.
- Other folders are borrowed from the official nuScenes dataset.

### Known Issues

- Nuscene ([issues-721](https://github.com/nutonomy/nuscenes-devkit/issues/721)) lacks translation in the z-axis, which makes it hard to recover accurate 6d localization and would lead to the misalignment of point clouds while accumulating them over whole scenes. Ground stratification occurs in several data.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

We provide a baseline model based on [BEVFormer](https://github.com/fundamentalvision/BEVFormer).

Please refer to [getting_started](docs/getting_started.md) for details.

<p align="right">(<a href="#top">back to top</a>)</p>

## Submission

### Submission site

Please submit your result on our [evaluation server](https://huggingface.co/spaces/AGC2024-S/occupancy-and-flow-2024). The submission rule can be referred to [here](#rules-for-occupancy-and-flow-challenge)

### Submission format

The submission must be a single `dict` with the following structure:

```
submission = {
    'method': 'XXXXX',
    'team': 'XXXXX',
    'authors': "XXXXX",
    'e-mail': "XXXXX",
    'institution / company': "XXXXXXXXXX",
    'country / region': "XXXXXXX",
    'results': {
        [sample_token1]: {
            'pcd_cls' (np.ndarray): predicted class ID of shape (N) and type np.uint8,
            'pcd_dist' (np.ndarray): predicted depth of shape (N) and type np.float16,
            'pcd_flow' (np.ndarray): predicted flow of shape (N, 2) and type np.float16,
        },
        [sample_token2]: {
            'pcd_cls' (np.ndarray): predicted class ID of shape (N) and type np.uint8,
            'pcd_dist' (np.ndarray): predicted depth of shape (N) and type np.float16,
            'pcd_flow' (np.ndarray): predicted flow of shape (N, 2) and type np.float16,
        },
        [sample_token3]: {
            'pcd_cls' (np.ndarray): predicted class ID of shape (N) and type np.uint8,
            'pcd_dist' (np.ndarray): predicted depth of shape (N) and type np.float16,
            'pcd_flow' (np.ndarray): predicted flow of shape (N, 2) and type np.float16,
        }
        ...
    }
}
```

Below is an example of how to save the submission:

```
open('submission.gz', 'wb').write(gzip.compress(pickle.dumps(submission), mtime=0))
```

We provide example scripts based on mmdetection3d to generate the submission file, please refer to [getting_started](docs/getting_started.md) for details.

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Before using the dataset, you should register on the website and agree to the terms of use of the [nuScenes](https://www.nuscenes.org/nuscenes).
All code within this repository is under [Apache 2.0 License](./LICENSE).

<p align="right">(<a href="#top">back to top</a>)</p>
