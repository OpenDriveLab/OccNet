<div align="center">   

<!-- omit in toc -->
# Scene as Occupancy
</div>

We believe **Occupancy** is a `general` representation of the scene and help perception and planning in the full-stack of autonomous driving. 

https://github.com/OpenDriveLab/OccNet/assets/54334254/92fb43a0-0ee8-4eab-aa53-0984506f0ec3



> **Scene as Occupancy**
> - [Paper in arXiv (to be released soon)]() | [CVPR 2023 AD Challenge Occupancy Track](https://opendrivelab.com/AD23Challenge.html#Track3)
> - Point of contact: [simachonghao@pjlab.org.cn](mailto:simachonghao@pjlab.org.cn) or [tongwenwen1@sensetime.com](mailto:tongwenwen1@sensetime.com). Any inquiries are welcome! 

<!-- omit in toc -->
## 3D Occupancy Leaderboard
We provide a full-scale 3D occupancy leaderboard based on our CVPR 2023 AD challenge, including their technical report. The link is to be launched! Stay tuned!


<!-- omit in toc -->
## Table of Contents
- [Highlights](#highlights)
- [News](#news)
- [Getting Started](#getting-started)
  - [Results and Pre-trained Models](#results-and-pre-trained-models)
- [TODO List](#todo-list)
- [License \& Citation](#license--citation)
- [Challenge](#challenge)
- [Related resources](#related-resources)

## Highlights

![teaser](assets/figs/pipeline.PNG)
- :oncoming_automobile: **General Representation in Perception**: 3D Occupancy is a geometry-aware representation of the scene. Compared to the form of 3D bounding box & BEV segmentation,  3D occupancy could capture the fine-grained details of critical obstacles in the scene.
- :trophy: **Exploration in full-stack Autonomous Driving**: OccNet, as a strong descriptor of the scene, could facilitate subsequent tasks such as perception and planning, achieving results on par with LiDAR-based methods (41.08 on mIOU in 3D occupancy, 60.46 on mIOU in LiDAR segmentation, 0.703 avg.Col in motion planning).

## News
- [2023/06/04] Code & model initial release `v1.0`
- [2023/06/04] 3D Occupancy and flow dataset release `v1.0`
- [2023/06/01] [CVPR AD Challenge 3D Occupancy Track](https://opendrivelab.com/AD23Challenge.html#Track3) close
- [2023/03/01] [CVPR AD Challenge 3D Occupancy Track](https://opendrivelab.com/AD23Challenge.html#Track3) launch

## Getting Started
- [Install](docs/install.md)
- [Prepare dataset](docs/prepare_dataset.md)
- [Visualization](docs/visualization.md)
- [Run the code](docs/run.md)

### Results and Pre-trained Models
We will release pre-trained weight soon.

![teaser](assets/figs/TABLE.png)


## TODO List
- [x] 3D Occupancy and flow dataset `v1.0`
- [x] 3D Occupancy Prediction code
- [ ] Pre-trained Models
- [ ] Occupancy label generation code
- [ ] Compatibility with other BEV encoders

## License & Citation
All assets (including figures and data) and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

Please consider citing our paper if the project helps your research with the following BibTex:
```bibtex
```

## Challenge
We host the first 3D occupancy prediciton challenge on CVPR 2023 End-to-end Autonomous Driving Workshop. For more information about the challenge, please refer to [here](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction).

## Related resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEV Perception Survey & Recipe](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe)
