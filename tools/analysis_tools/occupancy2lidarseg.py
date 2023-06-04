"""
transfer occupancy prediction result to lidarseg
lidar segmentation index 0 is ignored in the calculation.
"""

import mmcv   
import numpy as np   
import os
import os.path as osp                
from nuscenes import NuScenes 
import yaml
import sys               
from projects.mmdet3d_plugin.datasets.occupancy_metrics import SSCMetrics
from tqdm import tqdm

def  obtain_lidar_label_gt(lidar_seg_path):
    points_label = np.fromfile(lidar_seg_path, dtype=np.uint8)
    yaml_path = 'process_data/nuscenes_lidar_class.yaml'
    with open(yaml_path) as f:
        lidar_class_map = yaml.full_load(f)
    lidar_class_map = lidar_class_map['learning_map']
    points_label = np.vectorize(lidar_class_map.__getitem__)(points_label)  # 0->31  =>  0->16  0是空
    
    # 原始的lidar标签和occupancy不对齐，需要将二者进行对齐
    points_label = points_label - 1
    points_label[points_label ==-1] = occupancy_classes 
    points_label = lidarseg2occupancy_map[points_label]
    
    return points_label

def obtain_lidar_label_pred(occ_pred, points):
    
    valid_region = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[valid_region]    
    coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) / voxel_size[[2, 1, 0]]).astype(np.int64)
    z_coords, y_coords, x_coords = coords[:, 0], coords[:, 1], coords[:, 2]
    z_coords = np.clip(z_coords, 0, occ_zdim-1)
    y_coords = np.clip(y_coords, 0, occ_ydim-1)
    x_coords = np.clip(x_coords, 0, occ_xdim-1)
    
    occ_index = z_coords*occ_ydim*occ_xdim + y_coords*occ_xdim + x_coords
    occ_label = occ_pred[occ_index]
    return valid_region, occ_label 

def main(pkl_path, occupancy_pred_dir):

    data_infos = mmcv.load(pkl_path)['infos']
    eval_metrics = SSCMetrics(occupancy_classes, eval_far=False, eval_near=False)
    count = 0
    for index in tqdm(range(len(data_infos))):
        info = data_infos[index] 
        scene_name = info['scene_name']
        frame_idx = info['frame_idx'] 
        lidar_path = info['lidar_path']
        lidar_seg_path = info['lidar_seg_path']
        # load occ_pred
        occ_pred_path = os.path.join(occupancy_pred_dir, scene_name, 'occ_preds', '{:03d}_occ.npy'.format(frame_idx))
        occ_pred_sparse = np.load(occ_pred_path)
        occ_pred = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
        occ_pred[occ_pred_sparse[:, 0]] = occ_pred_sparse[:, 1]
        
        # load lidar label 
        lidar_label_gt = obtain_lidar_label_gt(lidar_seg_path)
        
        # get the predict lidar label from the occupancy result
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        points = points[:, :3]  # (n, 3)
        valid_region, lidar_label_pred = obtain_lidar_label_pred(occ_pred, points)
        lidar_label_gt = lidar_label_gt[valid_region]  # 只考虑特定范围内的点

        mask = np.ones(lidar_label_gt.shape[0]).astype(np.bool)
        mask[lidar_label_gt == occupancy_classes] = False  # 只考虑0-15这16类的gt
        mask[lidar_label_pred == occupancy_classes] = False
        
        lidar_label_gt[mask==False] = 255
        lidar_label_pred[mask==False] = 255
        
        y_pred = np.expand_dims(lidar_label_pred, axis=0)
        y_true = np.expand_dims(lidar_label_gt,axis=0)
        
        eval_metrics.add_batch(y_pred, y_true)
        count = count + 1
        
    print('sample count:', count)
    eval_resuslt = eval_metrics.get_stats()
    class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation']

    for i, class_name in enumerate(class_names):
        print("miou/{}: {:.2f}".format(class_name, eval_resuslt["iou_ssc"][i]))
    print("miou: {:.2f}".format(eval_resuslt["miou"]))
    print("iou: {:.2f}".format(eval_resuslt["iou"]))
    print("Precision: {:.4f}".format(eval_resuslt["precision"]))
    print("Recall: {:.4f}".format(eval_resuslt["recall"]))
        
    # foreground object
    print("foreground_iou: {:.2f}".format(eval_resuslt["foreground_iou"]))
    print("foreground_precision: {:.4f}".format(eval_resuslt["foreground_precision"]))
    print("foreground_recall: {:.4f}".format(eval_resuslt["foreground_recall"]))
    print("foreground_miou: {:.2f}".format(eval_resuslt["foreground_miou"]))


if __name__ == '__main__':  
    pkl_path = 'data/nuscenes_occupancy/nuscenes_infos_temporal_val_occ_gt.pkl'
    occupancy_pred_dir = 'exps/hybrid_tiny_occ/epoch_24/thre_0.25/'
    occ_type = 'normal'
    
    
    lidarseg_class_names=['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 
                'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation', 'empty'] 

    occupancy_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle','bicycle', 
                            'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
                            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'empty']

    lidarseg2occupancy_map = [] 
    for i in range(len(lidarseg_class_names)):  # lidarseg -> occupancy
        lidarseg2occupancy_map.append(occupancy_class_names.index(lidarseg_class_names[i]))
    lidarseg2occupancy_map = np.array(lidarseg2occupancy_map)

    pc_range = np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
    if occ_type == 'normal':
        voxel_size = np.array([0.5, 0.5, 0.5])  
    elif occ_type == 'coarse':
        voxel_size = np.array([1.0, 1.0, 1.0])
    occ_xdim = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    occ_ydim = int((pc_range[4] - pc_range[1]) / voxel_size[1])
    occ_zdim = int((pc_range[5] - pc_range[2]) / voxel_size[2])
    occupancy_classes = 16 
    voxel_num = occ_xdim*occ_ydim*occ_zdim
    
    main(pkl_path, occupancy_pred_dir)