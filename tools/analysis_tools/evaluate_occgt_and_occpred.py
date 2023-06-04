"""
detection-> occupancy
"""
import numpy as np
import os  
import os.path as osp 
from tqdm import tqdm
import sys
import shutil
import json 
from projects.mmdet3d_plugin.datasets.occupancy_metrics import SSCMetrics

CLASSES=('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
         'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
         'barrier')

occupancy_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle','bicycle', 
                         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

def main(occ_pred_dir, occ_gt_dir, consider_partial_scene=False):
    occupancy_classes = 10
    voxel_num = 640000
    near_distance = 25
    far_distance = 25
    eval_metrics = SSCMetrics(occupancy_classes, eval_far=True, eval_near=True,
                                     near_distance=near_distance, far_distance=far_distance)
    scene_names = os.listdir(occ_pred_dir)
    specific_scenes = []
    if consider_partial_scene:
        data_path = 'data/nuscenes_occupancy/val_dataset_irregular_scene.json'
        with open(data_path) as f:
            json_data = json.load(f)
        for key in json_data.keys():
            if json_data[key] > 200:
                specific_scenes.append(key)
    
    count = 0
    for scene_index in tqdm(range(len(scene_names))):
        scene_name = scene_names[scene_index]
        if consider_partial_scene and scene_name not in specific_scenes:
            continue
        occ_pred_scene_dir = osp.join(occ_pred_dir, scene_name)
        occ_gt_scene_dir = osp.join(occ_gt_dir, scene_name)
        frame_nums = len(os.listdir(osp.join(occ_pred_scene_dir, 'occ_preds')))//2
        for frame_idx in range(frame_nums):
            # gt 
            
            occ_gt_sparse = np.load(osp.join(occ_gt_scene_dir, 'occ_gt',  '{:03d}_occ_final.npy'.format(frame_idx)))
            occ_pred_sparse = np.load(osp.join(occ_pred_scene_dir, 'occ_preds', '{:03d}_occ.npy'.format(frame_idx)))
            
            # occ_gt
            occ_gt_index = occ_gt_sparse[:, 0]
            occ_gt_class = occ_gt_sparse[:, 1] 
            occ_gt_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
            occ_gt_occupancy[occ_gt_index] = occ_gt_class   
            
            # pred 
            occ_pred_index = occ_pred_sparse[:, 0]
            occ_pred_class = occ_pred_sparse[:, 1] 
            occ_pred_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
            occ_pred_occupancy[occ_pred_index] = occ_pred_class   

            # invalid region
            occ_invalid_index = np.load(os.path.join(occ_gt_scene_dir, 'occ_gt', '{:03d}_occ_invalid.npy'.format(frame_idx)))
            invalid_occupancy = np.ones(voxel_num, dtype=np.int64)
            invalid_occupancy[occ_invalid_index] = 255
            invalid_mask = np.expand_dims(invalid_occupancy, axis=0)
            
    
            y_pred = np.expand_dims(occ_pred_occupancy, axis=0)  # (1, 640000)
            y_true = np.expand_dims(occ_gt_occupancy, axis=0)
            eval_metrics.add_batch(y_pred, y_true, invalid=invalid_mask)
            count = count+1  

    print(f'======out evaluation metrics occupancy gt =========')
    eval_resuslt = eval_metrics.get_stats()
    
    for i, class_name in enumerate(CLASSES):
        print("miou/{}: {:.2f}".format(class_name, eval_resuslt["iou_ssc"][i]))
    print("miou: {:.2f}".format(eval_resuslt["miou"]))
    print("iou: {:.2f}".format(eval_resuslt["iou"]))
    print("Precision: {:.4f}".format(eval_resuslt["precision"]))
    print("Recall: {:.4f}".format(eval_resuslt["recall"]))
    
    
    if eval_resuslt['far_metrics'] is not None:
        far_metrics_dict = eval_resuslt['far_metrics']
        print('')
        print('far distance:', far_distance)
        for key in far_metrics_dict:
            if key!= 'far_iou_ssc':
                print("{}: {:.2f}".format(key, far_metrics_dict[key]))
            else:
                for i, class_name in enumerate(CLASSES):
                    print("far_miou/{}: {:.2f}".format(class_name, far_metrics_dict[key][i]))
    
    if eval_resuslt['near_metrics'] is not None:
        print('')
        print('near distance:', near_distance)
        near_metrics_dict = eval_resuslt['near_metrics']
        for key in near_metrics_dict:
            if key!= 'near_iou_ssc':
                print("{}: {:.2f}".format(key, near_metrics_dict[key]))
            else:
                for i, class_name in enumerate(CLASSES):
                    if i < 10:
                        print("near_miou/{}: {:.2f}".format(class_name, near_metrics_dict[key][i]))
    
    
if __name__ == '__main__':

    occ_gt_dir = 'data/nuscenes_occupancy/val'
    occ_pred_dir = 'exps/hybrid_tiny_occ_block5_v2_1/epoch_24/thre_0.25'
    
    print('occ_pred_dir:', occ_pred_dir)
    consider_partial_scene = True 
    print('consider_partial_scene:', consider_partial_scene)
    
    main(occ_pred_dir, occ_gt_dir, consider_partial_scene)
   
