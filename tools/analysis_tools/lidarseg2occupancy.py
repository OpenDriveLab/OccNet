"""
cylinder3d预测lidarseg结果后转换为occupancy，与occupancy GT进行对比
"""
import numpy as np
import os  
import os.path as osp 
from tqdm import tqdm
import sys
import shutil
from projects.mmdet3d_plugin.datasets.occupancy_metrics import SSCMetrics


lidaroccupancy_dir = '/mnt/lustre/tongwenwen1/codes/Cylinder3D/inference/exp1/occupancy'  # lidarinfer转换的结果
gtoccupancy_dir= '/mnt/lustre/share_data/tongwenwen1/public_datasets/nuscenes_occupancy/val'

occupancy_classes = 16 
voxel_num = 200*200*16

eval_metrics = SSCMetrics(occupancy_classes)

count = 0
for scene in sorted(os.listdir(lidaroccupancy_dir)):
    print('process scene:', scene)
    frame_num = len(os.listdir(osp.join(lidaroccupancy_dir, scene)))
    for frame_id in range(frame_num):
        lidar_occ = np.load(osp.join(lidaroccupancy_dir, scene, '{:03d}_occ.npy'.format(frame_id)))  # 不包含空的点
        gt_occ = np.load(osp.join(gtoccupancy_dir, scene, 'occ_gt', '{:03d}_occ_final.npy'.format(frame_id)))
        
        # load gt 
        occ_index = gt_occ[:, 0]
        occ_class = gt_occ[:, 1] 
        gt_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
        gt_occupancy[occ_index] = occ_class  # (num_voxels)
        
        # load pred
        occ_index = lidar_occ[:, 0]
        occ_class = lidar_occ[:, 1] 
        lidar_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
        lidar_occupancy[occ_index] = occ_class  # (num_voxels)
        
        # mask the region where lidar occ does not exist
        mask = np.zeros(voxel_num).astype(np.bool)
        mask[occ_index] = True
        
        gt_occupancy[mask==False] = 255
        lidar_occupancy[mask==False] = 255
        
        y_pred = np.expand_dims(lidar_occupancy, axis=0)
        y_true = np.expand_dims(gt_occupancy,axis=0)
        
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

if eval_resuslt['far_metrics'] is not None:
    far_metrics_dict = eval_resuslt['far_metrics']
    for key in far_metrics_dict:
        if key!= 'far_iou_ssc':
            print("{}: {:.2f}".format(key, far_metrics_dict[key]))
        else:
            for i, class_name in enumerate(class_names):
                if i < 10:
                    print("far_miou/{}: {:.2f}".format(class_name, far_metrics_dict[key][i]))
