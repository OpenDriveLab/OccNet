import mmcv
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from mmdet3d.datasets import NuScenesDataset
import os  
import os.path as osp 
from tqdm import tqdm
import sys
import shutil
from projects.mmdet3d_plugin.datasets.occupancy_metrics import SSCMetrics

CLASSES=('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
occ_size = 0.5
voxel_size = np.array([0.5, 0.5, 0.5])
occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
occupancy_classes = 10
voxel_num = occ_xdim*occ_ydim*occ_zdim

def parse_single_sample(pred_info, gt_info, thred_score=0.2):
    # box in the global system
    pred_boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                         name=record['detection_name'], token='predicted') for record in pred_info
                         if record['detection_score'] > thred_score]
    # print('pred boxes number', len(pred_boxes))
    pred_boxes_list = []
    for box in pred_boxes:
        # revise box to lidar coordinate
        box.translate(-np.array(gt_info['ego2global_translation']))
        box.rotate(Quaternion(gt_info['ego2global_rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(gt_info['lidar2ego_translation']))
        box.rotate(Quaternion(gt_info['lidar2ego_rotation']).inverse)
        pred_boxes_list.append(box)  # now the box is in the lidar coordinate system
    
    locs = np.array([b.center for b in pred_boxes_list]).reshape(-1, 3)
    dims = np.array([b.wlh for b in pred_boxes_list]).reshape(-1, 3)
    dims_lwh = np.concatenate([dims[:, 1:2], dims[:, 0:1], dims[:, 2:]], axis=-1)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                        for b in pred_boxes_list]).reshape(-1, 1)
    boxes = np.concatenate([locs, dims_lwh, rots], axis=1)
    names = [b.name for b in pred_boxes_list]
    for i in range(len(names)):
        if names[i] in NuScenesDataset.NameMapping:
            names[i] = NuScenesDataset.NameMapping[names[i]]

    pred_info = {}
    pred_info['gt_boxes_st'] = boxes
    pred_info['gt_names'] = names
    pred_occupancy = box2occ(pred_info)
    return pred_occupancy

def box2occ(info):
    boxes  = info['gt_boxes_st']
    names = info['gt_names']
    labels = []
    for cat in names:
        if cat in CLASSES:
            labels.append(CLASSES.index(cat))
        else:
            labels.append(-1)
    labels = np.array(labels)
    mask = labels >= 0    # abandon class with label = -1
    boxes = boxes[mask] 
    labels = labels[mask]
    
    occ_index_set = set()
    occ_indexs = []
    occ_labels = []
    for box_num in range(len(boxes)): 
        box = boxes[box_num]
        label = labels[box_num]
        # calculate the occupancy of box in the 3D discrete space 
        # sample point from the box
        xc, yc, zc, length, width, height, theta = box 
        nx = int(length/occ_size*2)+10
        ny = int(width/occ_size*2)+10
        nz = int(height/occ_size*2)+10
        loc_x = np.linspace(-length/2-occ_size, length/2+occ_size, nx)
        loc_y = np.linspace(-width/2-occ_size, width/2+occ_size, ny)
        loc_z = np.linspace(-height/2-occ_size, height/2+occ_size, nz)
        x, y, z = np.meshgrid(loc_x, loc_y, loc_z, indexing='ij')

        loc_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        mask = np.ones(loc_points.shape[0]).astype(np.bool)
        fill_scale = 0.2
        mask = np.logical_and(mask, abs(loc_points[:, 0]) <= (length+fill_scale)/2)
        mask = np.logical_and(mask, abs(loc_points[:, 1]) <= (width+fill_scale)/2)
        mask = np.logical_and(mask, abs(loc_points[:, 2]) <= (height+fill_scale)/2)
        
        loc_points = loc_points[mask]
    
        bottom = np.ones((loc_points.shape[0], 1))
        loc_points_hom = np.concatenate([loc_points, bottom], axis=-1).T

        box2lidar = np.eye(4, 4)
        box2lidar[:3, -1] = np.array([xc, yc, zc])
        box2lidar[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])
        
        points = box2lidar @ loc_points_hom
        points = (points.T)[:, :3]

        pc_range = np.array(point_cloud_range)
        keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
                (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        points = points[keep, :]    
        coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) / voxel_size[[2, 1, 0]]).astype(np.int64)
        z_coords, y_coords, x_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        z_coords = np.clip(z_coords, 0, occ_zdim-1)
        y_coords = np.clip(y_coords, 0, occ_ydim-1)
        x_coords = np.clip(x_coords, 0, occ_xdim-1)
        coords = np.stack((z_coords, y_coords, x_coords), axis=-1)
        coords = np.unique(coords, axis=0)
        
        occ_global_index = occ_ydim*occ_xdim*coords[:, 0]+occ_xdim*coords[:, 1]+coords[:, 2]
        
        cur_occ_index_list = []
        for i in range(len(occ_global_index)):
            if occ_global_index[i] in occ_index_set:
                continue
            else:
                cur_occ_index_list.append(occ_global_index[i])
        # print(len(cur_occ_index_list))
        occ_index_set.union(set(cur_occ_index_list))
        occ_indexs.extend(cur_occ_index_list)
        occ_labels.extend([label]*len(cur_occ_index_list))
    
    occ_indexs =np.array(occ_indexs, dtype=np.int64).reshape(-1, 1)
    occ_labels = np.array(occ_labels, dtype=np.int64).reshape(-1, 1)

    occ_index_label = np.concatenate([occ_indexs, occ_labels], axis=-1)
    # print(occ_index_label.shape)

    return occ_index_label
            
def main(detection_results, gt_infos, pred_save_dir, gt_save_dir, recalculate_gt, thred_score=0.2):
    gt_infos_dict = {}
    for gt_info in gt_infos:
        sample_token = gt_info['token']
        gt_infos_dict[sample_token] = gt_info
    
    assert set(detection_results.keys()) == set(gt_infos_dict.keys())
    
    save_scene_names=['scene-0099', 'scene-0105', 'scene-0276', 'scene-0626']
    sample_tokens = list(gt_infos_dict.keys())
    eval_metrics = SSCMetrics(occupancy_classes)
    for index in tqdm(range(len(sample_tokens))):
        sample_token = sample_tokens[index]
        pred_info = detection_results[sample_token]
        gt_info = gt_infos_dict[sample_token]
        scene_name = gt_info['scene_name']
        frame_idx = gt_info['frame_idx']
        # if scene_name != 'scene-0099':
        #     continue
        
        occ_pred_sparse = parse_single_sample(pred_info, gt_info, thred_score=thred_score)
        
        if recalculate_gt:
            occ_gt_sparse = box2occ(gt_info)
            gt_scene_dir = os.path.join(gt_save_dir, scene_name)
            occ_gt_save_dir = os.path.join(gt_scene_dir, 'occ_gts')
            os.makedirs(occ_gt_save_dir, exist_ok=True)
            
            image_save_dir = os.path.join(gt_scene_dir, 'images')
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = osp.join(image_save_dir, '{:05d}.png'.format(frame_idx))
            if 'surround_image_path' in gt_info:
                shutil.copyfile(gt_info['surround_image_path'], image_save_path)
            np.save(osp.join(occ_gt_save_dir, '{:05d}_occ.npy'.format(frame_idx)), occ_gt_sparse)  
        
        else:
            occ_gt_sparse_path = osp.join(gt_save_dir, scene_name, 'occ_gts', '{:05d}_occ.npy'.format(frame_idx))
            occ_gt_sparse = np.load(occ_gt_sparse_path)
        
        # gt 
        occ_index = occ_gt_sparse[:, 0]
        occ_class = occ_gt_sparse[:, 1] 
        gt_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
        gt_occupancy[occ_index] = occ_class
            
        # pred 
        occ_index = occ_pred_sparse[:, 0]
        occ_class = occ_pred_sparse[:, 1] 
        pred_occupancy = np.ones(voxel_num, dtype=np.int64)*occupancy_classes
        pred_occupancy[occ_index] = occ_class
        
        y_pred = np.expand_dims(pred_occupancy, axis=0)  # (1, 640000)
        y_true = np.expand_dims(gt_occupancy, axis=0)
        eval_metrics.add_batch(y_pred, y_true)
        
        pred_scene_dir = os.path.join(pred_save_dir, scene_name)
        occ_pred_save_dir = os.path.join(pred_scene_dir, 'occ_preds')
        os.makedirs(occ_pred_save_dir, exist_ok=True)
        np.save(osp.join(occ_pred_save_dir, '{:05d}_occ.npy'.format(frame_idx)), occ_pred_sparse)  
        
    print(f'======out evaluation metrics =========')
    eval_resuslt = eval_metrics.get_stats()
    for i, class_name in enumerate(CLASSES):
        print("miou/{}: {:.2f}".format(class_name, eval_resuslt["iou_ssc"][i]))
    print("miou: {:.2f}".format(eval_resuslt["miou"]))
    print("iou: {:.2f}".format(eval_resuslt["iou"]))
    print("Precision: {:.4f}".format(eval_resuslt["precision"]))
    print("Recall: {:.4f}".format(eval_resuslt["recall"]))
    
    if eval_resuslt['far_metrics'] is not None:
        far_metrics_dict = eval_resuslt['far_metrics']
        for key in far_metrics_dict:
            if key!= 'far_iou_ssc':
                print("{}: {:.2f}".format(key, far_metrics_dict[key]))
            else:
                for i, class_name in enumerate(CLASSES):
                    print("far_miou/{}: {:.2f}".format(class_name, far_metrics_dict[key][i]))

if __name__ == '__main__':
    # bevformer_tiny: only detection
    
    data_dir = 'val/exps/bev_tiny_det_v2_deterministic_exp2/Sun_Jan_29_14_51_19_2023/pts_bbox'
    exp_name =  'bev_tiny_det_v2_deterministic_exp2'
    recalculate_gt = True
    data_path = os.path.join(data_dir, 'results_nusc.json')
    detection_results = mmcv.load(data_path)['results']
    
    # scene_token_path = 'data/nuscenes_occupancy/validation_scene_tokens.json'
    # scene_token_infos = mmcv.load(scene_token_path) 
    
    pkl_path = 'data/nuscenes_occupancy/nuscenes_infos_temporal_val_occ_gt.pkl'
    gt_infos = mmcv.load(pkl_path)['infos']
    
    thred_score = 0.2
    save_dir = 'results/detection2occupancy'
    os.makedirs(save_dir, exist_ok=True)
    
    pred_save_dir = os.path.join(save_dir, exp_name, 'score_'+str(thred_score))
    os.makedirs(pred_save_dir, exist_ok=True)
    
    gt_save_dir = os.path.join(save_dir, 'boxgt')
    os.makedirs(gt_save_dir, exist_ok=True)
    
    main(detection_results, gt_infos, pred_save_dir, gt_save_dir, recalculate_gt, thred_score=thred_score)
   
