# Prepare nuScenes 3D Detection and 3D Occupancy Data
**1. Download nuScenes V1.0 full dataset and can bus data [HERE](https://www.nuscenes.org/download). Organize the folder structure:**
```
OccNet
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
```

**2. Prepared nuScenes 3D detection data**
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```
Using the above code will generate the following files
`
data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl
`
**3. Prepare 3D Occupancy dataset**
> Dataset Information
<div align="left">
  
| Type |  Info |
| :----: | :----: |
| train           | 28,130 |
| val             | 6,019 |
| cameras         | 6 |
| voxel size      | 0.5m |
| range           | [-50m, -50m, -5m, 50m, 50m, 3m]|
| volume size     | [200, 200, 16]|
| #classes        | 0 - 15 |
</div>

> Dowload link of train and validation dataset

| Version | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Size |
| :---: | :---: | :---: | :---: |
| trainval  | TBD | TBD | - |


> Merge 3D detection and 3D occupancy dataset
```
python tools/create_data_with_occ.py
```
