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

def main(box_gt_dir, box_det_dir, occ_gt_dir, consider_partial_scene=False, occ_type='coarse'):
    
    occupancy_classes = 10
    if occ_type == 'coarse':
        voxel_num = 640000
    else:
        voxel_num = 400*400*32
    near_distance = 25
    far_distance = 25
    eval_metrics_occ_gt = SSCMetrics(occupancy_classes, eval_far=True, eval_near=True,
                                     near_distance=near_distance, far_distance=far_distance,
                                     occ_type=occ_type)
    eval_metrics_box_gt = SSCMetrics(occupancy_classes, eval_far=True, eval_near=True,
                                     near_distance=near_distance, far_distance=far_distance,
                                     occ_type=occ_type)
    scene_names = os.listdir(box_gt_dir)
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
        box_gt_scene_dir = osp.join(box_gt_dir, scene_name)
        box_pred_scene_dir = osp.join(box_det_dir, scene_name)
        occ_gt_scene_dir = osp.join(occ_gt_dir, scene_name)
        if occ_type == 'coarse':
            frame_nums = len(os.listdir(osp.join(box_pred_scene_dir, 'occ_preds')))
        else:
            frame_nums = len(os.listdir(osp.join(box_pred_scene_dir, 'occ_preds_fine')))
        
        for frame_idx in range(frame_nums):
            # gt 
            
            occ_gt_sparse = np.load(osp.join(occ_gt_scene_dir, 'occ_gt',  '{:03d}_occ_final.npy'.format(frame_idx)))
            box_gt_sparse = np.load(osp.join(box_gt_scene_dir,  'occ_gts', '{:05d}_occ.npy'.format(frame_idx)))
            if occ_type == 'coarse':
                box_pred_sparse = np.load(osp.join(box_pred_scene_dir, 'occ_preds', '{:05d}_occ.npy'.format(frame_idx)))
            else:
                box_pred_sparse = np.load(osp.join(box_pred_scene_dir, 'occ_preds_fine', '{:05d}_occ.npy'.format(frame_idx)))
            
            # occ_gt
            occ_gt_index = occ_gt_sparse[:, 0]
            occ_gt_class = occ_gt_sparse[:, 1] 
            occ_gt_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
            occ_gt_occupancy[occ_gt_index] = occ_gt_class   
            
            # box_gt 
            box_gt_index = box_gt_sparse[:, 0]
            box_gt_class = box_gt_sparse[:, 1] 
            box_gt_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
            box_gt_occupancy[box_gt_index] = box_gt_class   

            # pred 
            box_pred_index = box_pred_sparse[:, 0]
            box_pred_class = box_pred_sparse[:, 1] 
            box_pred_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
            box_pred_occupancy[box_pred_index] = box_pred_class   

            # invalid region
            occ_invalid_index = np.load(os.path.join(occ_gt_scene_dir, 'occ_gt', '{:03d}_occ_invalid.npy'.format(frame_idx)))
            invalid_occupancy = np.ones(voxel_num, dtype=np.int64)
            invalid_occupancy[occ_invalid_index] = 255
            invalid_mask = np.expand_dims(invalid_occupancy, axis=0)
            
    
            y_pred = np.expand_dims(box_pred_occupancy, axis=0)  # (1, 640000)
            y_true = np.expand_dims(occ_gt_occupancy, axis=0)
            eval_metrics_occ_gt.add_batch(y_pred, y_true, invalid=invalid_mask)
            
            y_true = np.expand_dims(box_gt_occupancy, axis=0)
            eval_metrics_box_gt.add_batch(y_pred, y_true, invalid=invalid_mask)
            
            count = count+1  


    print("far_miou/{}: {:.2f}".format(class_name, far_metrics_dict[key][i]))
    # print('')
    print(f'======out evaluation metrics occupancy gt =========')
    eval_resuslt = eval_metrics_occ_gt.get_stats()
    
    for i, class_name in enumerate(CLASSES):
        print("miou/{}: {:.2f}".format(class_name, eval_resuslt["iou_ssc"][i]))
    print("miou: {:.2f}".format(eval_resuslt["miou"]))
    print("iou: {:.2f}".format(eval_resuslt["iou"]))
    print("Precision: {:.4f}".format(eval_resuslt["precision"]))
    print("Recall: {:.4f}".format(eval_resuslt["recall"]))
    
    
    if eval_resuslt['far_metrics'] is not None:
        print('')
        print('far distance:', far_distance)
        far_metrics_dict = eval_resuslt['far_metrics']
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
    # occ_type = 'fine'  # or fine
    occ_type = 'coarse' 
    if occ_type == 'coarse':
        occ_gt_dir = 'data/nuscenes_occupancy/val'
        box_gt_dir = 'results/detection2occupancy/boxgt'
    else:
        occ_gt_dir = 'data/nuscenes_occupancy_fine/val'
        box_gt_dir = 'results/detection2occupancy/boxgt_fine'
    box_det_dir = 'results/detection2occupancy/hybrid_tiny_det_occ_block5_v2/score_0.2'
    consider_partial_scene = True 
    
    print('box_dir:', box_det_dir)
    print('consider_partial_scene:', consider_partial_scene)
    main(box_gt_dir, box_det_dir, occ_gt_dir, consider_partial_scene, occ_type)
   
