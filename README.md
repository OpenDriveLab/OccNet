<div id="top" align="center">

# CVPR 2024 3D Occupancy and Flow Challenge

<a href="#devkit">
  <img alt="devkit: v0.1.0" src="https://img.shields.io/badge/devkit-v0.1.0-blueviolet"/>
</a>
<a href="#license">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>

<img src="./figs/occupanc_1.gif" width="696px">

</div>

## Introduction

Understanding the 3D surroundings including the background stuffs and foreground objects is important for autonomous driving. In the traditional 3D object detection task, a foreground object is represented by the 3D bounding box. However, the geometrical shape of the object is complex, which can not be represented by a simple 3D box, and the perception of the background stuffs is absent. The goal of this task is to predict the 3D occupancy of the scene. In this task, we provide a large-scale occupancy benchmark based on the nuScenes dataset. The benchmark is a voxelized representation of the 3D space, and the occupancy state and semantics of the voxel in 3D space are jointly estimated in this task. The complexity of this task lies in the dense prediction of 3D space given the surround-view images.

If you use the challenge dataset in your paper, please consider citing OccNet and Occ3D with the following BibTex:

```bibtex
@article{sima2023_occnet,
      title={Scene as Occupancy},
      author=author={Chonghao Sima and Wenwen Tong and Tai Wang and Li Chen and Silei Wu and Hanming Deng and Yi Gu and Lewei Lu and Ping Luo and Dahua Lin and Hongyang Li},
      year={2023},
      eprint={2306.02851},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Table of Contents
- [CVPR 2024 Occupancy and Flow Challenge](#cvpr-2024-occupancy-and-flow-challenge)
  - [Introduction](#introduction)
  - [Task Definition](#task-definition)
    - [Rules for Occupancy Challenge](#rules-for-occupancy-challenge)
  - [Evaluation Metrics](#evaluation-metrics)
    - [mIoU](#miou)
    - [F Score](#f-score)
  - [Data](#data)
    - [Basic Information](#basic-information)
    - [Download](#download)
    - [Hierarchy](#hierarchy)
    - [Known Issues](#known-issues)
  - [Getting Started](#getting-started)
  - [Submission](#submission)
  - [Timeline](#challenge-timeline)
  - [License](#license)

## Task Definition

Given images from multiple cameras, the goal is to predict the current occupancy state and semantics of each voxel grid in the scene.

### Rules for Occupancy Challenge

* We allow using annotations provided in the nuScenes dataset, and during inference, the input modality of the model should be camera only. 
* No future frame is allowed during inference.
* In order to check the compliance, we will ask the participants to provide technical reports to the challenge committee and the participant will be asked to provide a public talk about the method after winning the award.
* Every submission provides method information. We encourage publishing code, but do not make it a requirement.
* Each team can have at most one account on the evaluation server. Users that create multiple accounts to circumvent the rules will be excluded from the challenge.
* Each team can submit at most three results during the challenge. 
* Faulty submissions that return an error on Eval AI do not count towards the submission limit.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from the challenge.

<p align="right">(<a href="#top">back to top</a>)</p>

## Evaluation Metrics

Leaderboard ranking for this challenge is by the intersection-over-union (mIoU) over all classes. 

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

### Occupancy Flow

Here we measure velocity errors for a set of true positives (TP). We use a threshold of 2m distance.

The absolute velocity error is defined for 8 classes ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian') in m/s. 


### Occupancy Score

The final occupancy score is defined to be a weighted sum of mIoU and mAVE. Note that the velocity errors are converted to velocity scores as `max(1 - AVE, 0.0)`. That is,

```
OccScore = mIoU * 0.9 + max(1 - AVE, 0.0) * 0.1
```

<p align="right">(<a href="#top">back to top</a>)</p>

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

2. Download our `openocc_v2.zip` and `infos.zip` from [HERE](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq)

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

Coming soon.

<p align="right">(<a href="#top">back to top</a>)</p>

## License
Before using the dataset, you should register on the website and agree to the terms of use of the [nuScenes](https://www.nuscenes.org/nuscenes).
All code within this repository is under [MIT License](./LICENSE).

<p align="right">(<a href="#top">back to top</a>)</p>
