# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
"""
generate the occ gt frame by frame

(1) voxelize the 3D space
(2) classifiy the voxel based on 3D box and LiDAR segmentation: 10 foreground + 6 background

output: occupancy + flow
"""
import os
import os.path as osp
import numpy as np 
import mmcv
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from torch.nn import functional as F
from collections import defaultdict
import copy
from utils import *
import numba as nb

class SingleFrameOCCGTGenerator():
    """save the gt occupancy 
    """
    def __init__(self, info, scene_dir, train_flag=False, occ_resolution='normal', 
                 postprocess=False, voxel_point_threshold=0,
                 save_flow_info=True):
        self.info = info
        self.use_valid_flag = False
        self.CLASSES=('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
        self.with_velocity = True
        self.train_flag = train_flag
        self.scene_dir = scene_dir  # TODO 
        self.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        self.occ_resolution = occ_resolution
        self.voxel_point_threshold = voxel_point_threshold
        self.save_flow_info = save_flow_info
        if self.occ_resolution == 'normal':
            self.occupancy_size = [0.5, 0.5, 0.5]
        elif self.occ_resolution == 'fine':
            self.occupancy_size = [0.25, 0.25, 0.25]
        elif self.occ_resolution == 'coarse':
            self.occupancy_size = [1.0, 1.0, 1.0]
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.occupancy_classes = 16  # 0-9 foreground, 10-15 background
        self.background_label_map = {11:10, 12:11, 13:12, 14:13, 15:14, 16:15, 0: 255}  # 0 is noise
        self.background_labels = {10, 11, 12, 13, 14, 15}
        self.enlarge_box = self.occupancy_size[-1]
        self.postprocess = postprocess  
        self.img_w = 1600
        self.img_h = 900
        self.background_label = self.occupancy_classes - 1

    def obtain_sample_background_voxel(self, lidarseg_points, lidarseg_labels):
        # label of background lidar point
        lidarseg_velocity = np.zeros((lidarseg_labels.shape[0], 2))

        lidarseg_occ, _ = self.obtain_occ_gt(lidarseg_points, lidarseg_labels, lidarseg_velocity)
        return lidarseg_occ

    def postprocess_generated_occ_gt(self, occ_index_label, occ_velocitys):
        occ_index_label, occ_velocitys = self.remove_isolated_voxel(occ_index_label, occ_velocitys)
        occ_index_label, occ_velocitys = self.remove_segments_in_air(occ_index_label, occ_velocitys)
        occ_index_label, occ_velocitys = self.fill_empty_voxel(occ_index_label, occ_velocitys)

    def index1d_2_index3d(self, index1d):
        index_z = index1d//(self.occ_xdim*self.occ_ydim)
        index_y = (index1d - index_z*self.occ_xdim*self.occ_ydim)//self.occ_xdim
        index_x = index1d % self.occ_xdim
        index3d = (index_z, index_y, index_x)
        return index3d

    def index3d_2_index1d(self, index3d):
        index_z, index_y, index_x = index3d
        index1d = index_z*self.occ_xdim*self.occ_ydim + index_y*self.occ_xdim+ index_x
        return index1d

    def map_sample2sweep(self, lidarseg_back_occ, occ_index_label, occ_velocity):
        """ determine the label of sweep data based on sample backgorund data"""
        occ_index, occ_label = occ_index_label[:, 0], occ_index_label[:, 1]
        sample_occ_index, sample_occ_label = lidarseg_back_occ[:, 0], lidarseg_back_occ[:, 1]

        result_dict = defaultdict(dict)
        """
        0. noise points: remove noise
        """
        sample_noise_index_set = set()
        sample_occ_index_label_dict = {}
        for i in range(len(sample_occ_label)):
            if sample_occ_label[i] == 0:
                sample_noise_index_set.add(sample_occ_index[i])
            sample_occ_index_label_dict[sample_occ_index[i]] = sample_occ_label[i]
        
        occ_mask = np.ones_like(occ_index).astype(np.bool)
        for i in range(len(occ_index)):
            if occ_index[i] in sample_noise_index_set:
                occ_mask[i] = False
        occ_index = occ_index[occ_mask]
        occ_label = occ_label[occ_mask]
        occ_velocity = occ_velocity[occ_mask]
        """
        1. map sample_occ_index to occ_index 
        """

        for i in range(len(occ_index)):
            if occ_label[i] == self.occupancy_classes and occ_index[i] in sample_occ_index_label_dict:
                occ_label[i] = self.background_label_map[sample_occ_index_label_dict[occ_index[i]]]

        temp_occ_index_label = np.stack([occ_index, occ_label], axis=-1)
        temp_occ_velocity = copy.deepcopy(occ_velocity)
        result_dict['step1']['occ'] = temp_occ_index_label
        result_dict['step1']['flow'] = temp_occ_velocity
        """
        2. fill unknow background occ_label: label = self.occupancy_classes = 16
        """
        growth_step = 50
        map_dict = {}  # index1d -> label 
        occ_index_sets = set()
        for i in range(len(occ_index)):
            if occ_label[i] != self.occupancy_classes:
                map_dict[occ_index[i]] = occ_label[i]
                occ_index_sets.add(occ_index[i])
        
        for i in range(growth_step):
            unknown_mask = occ_label == self.occupancy_classes
            unknown_occ_index = occ_index[unknown_mask]
            unknown_occ_label = occ_label[unknown_mask]
            count = 0
            for i in range(len(unknown_occ_index)):
                index1d = unknown_occ_index[i]
                cur_label = self.estimate_background_occ_status(index1d, occ_index_sets, map_dict)
                if cur_label is not None:
                    unknown_occ_label[i] = cur_label
                    count = count+1
            # update 
            occ_index[unknown_mask] = unknown_occ_index
            occ_label[unknown_mask] = unknown_occ_label
            for i in range(len(unknown_occ_index)):
                if unknown_occ_label[i] != self.occupancy_classes:
                    occ_index_sets.add(unknown_occ_index[i])
                    map_dict[unknown_occ_index[i]] = unknown_occ_label[i]
        
        temp_occ_index_label = np.stack([occ_index, occ_label], axis=-1)
        temp_occ_velocity = copy.deepcopy(occ_velocity)
        result_dict['step2']['occ'] = temp_occ_index_label
        result_dict['step2']['flow'] = temp_occ_velocity

        """
        3. remove uncertain points
        """
        known_mask = occ_label != self.occupancy_classes
        occ_index = occ_index[known_mask]
        occ_label = occ_label[known_mask]
        occ_velocity = occ_velocity[known_mask]

        temp_occ_index_label = np.stack([occ_index, occ_label], axis=-1)
        temp_occ_velocity = copy.deepcopy(occ_velocity)
        result_dict['step3']['occ'] = temp_occ_index_label
        result_dict['step3']['flow'] = temp_occ_velocity


        """
        4. fill holes in the space
        """
        occ_index_label = np.stack([occ_index, occ_label], axis=-1)
        occ_index_label, occ_velocity = self.fill_empty_voxel(occ_index_label, occ_velocity)
        result_dict['final']['occ'] = occ_index_label
        result_dict['final']['flow'] = occ_velocity

        return result_dict
    
    def estimate_background_occ_status(self, index1d, occ_index_sets, map_dict):
        index3d = self.index1d_2_index3d(index1d)
        neighbors = [[0, -1, 0], [0, 1, 0],
                     [0, 0, -1], [0, 0, 1],
                     [0, -1, 1], [0, 1, 1],
                     [0, -1, -1], [0, 1, -1]
                    ]
        
        labels = []
        for neighbor in neighbors:
            neighbor_index_3d = (index3d[0]+neighbor[0], index3d[1]+neighbor[1], index3d[2]+neighbor[2])
            neighbor_index_1d = self.index3d_2_index1d(neighbor_index_3d)
            if neighbor_index_1d in occ_index_sets:
                label = map_dict[neighbor_index_1d]
                if label in self.background_labels:
                    labels.append(label)
        
        if len(labels):
            vote_label = np.argmax(np.bincount(labels))
            return vote_label
        else:
            return None
        
    def save_occ_gt(self, points, lidarseg_points, lidarseg_labels):
        
        info = self.info
        scene_name = info['scene_name']
        frame_idx = info['frame_idx']
        print(f"scene_name {scene_name} frame_idx {frame_idx}")
  
        # sample: background occ
        lidarseg_back_occ = self.obtain_sample_background_voxel(lidarseg_points, lidarseg_labels)

        # sample+sweep
        points, point_label, points_velocity = self.split_lidar_with_bbox(points) 
        occ_index_label, occ_velocitys = self.obtain_occ_gt(points, point_label, points_velocity)

        result_dict = self.map_sample2sweep(lidarseg_back_occ, occ_index_label, occ_velocitys)
        self.occ_gt_dir = os.path.join(self.scene_dir, 'occ_gt')
        os.makedirs(self.occ_gt_dir, exist_ok=True)

        occ_save_path = os.path.join(self.occ_gt_dir, '{:03d}_occ.npy'.format(frame_idx))  
        flow_save_path = os.path.join(self.occ_gt_dir, '{:03d}_flow.npy'.format(frame_idx))  
        np.save(occ_save_path, result_dict['final']['occ'])  
        if self.save_flow_info:
            np.save(flow_save_path, result_dict['final']['flow']) 
        
        # obtain invalid mask for invalid data
        # if self.train_flag == False:
        #     self.camera_view_mask()
        self.camera_view_mask()

        occ_gt_path = os.path.join(self.occ_gt_dir, '{:03d}_occ.npy'.format(frame_idx))
        flow_gt_path = os.path.join(self.occ_gt_dir, '{:03d}_flow.npy'.format(frame_idx))
        occ_invalid_path = os.path.join(self.occ_gt_dir, '{:03d}_occ_invalid.npy'.format(frame_idx))

        return occ_gt_path, flow_gt_path, occ_invalid_path

    def camera_view_mask(self):
        """
        set the camera invisible voxel as invalid voxel
        """
        info = self.info
        frame_idx = info['frame_idx']
        occ_gt_path = os.path.join(self.occ_gt_dir, '{:03d}_occ.npy'.format(frame_idx))
        occ_gt = np.load(occ_gt_path) 

        # classify occ gt into surface and occluded occ
        cam_occ_gt_pixels_dict, occ_gt_index_surfaces = self.split_occ_gt_into_surface_occluded(occ_gt, info)
        occ_gt_index_surfaces = np.array(list(occ_gt_index_surfaces))

        invalid_occ_indexs = self.obtain_invalid_occ(occ_gt, info, cam_occ_gt_pixels_dict)
        invalid_occ_path = os.path.join(self.occ_gt_dir, '{:03d}_occ_invalid.npy'.format(frame_idx))
        np.save(invalid_occ_path, invalid_occ_indexs)

    def obtain_invalid_occ(self, occ_gt, info, cam_occ_gt_pixels_dict):
        occ_gt_index = occ_gt[:, 0]
        occ_gt_label = occ_gt[:, 1]
        temp_x = np.arange(self.occ_xdim)
        temp_y = np.arange(self.occ_ydim)
        temp_z = np.arange(self.occ_zdim)
        invalid_occ_indexs = set()
        
        for cam_type in info['cams']:  # 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            cam_info = info['cams'][cam_type]
            cam_intrinsic = cam_info['cam_intrinsic']
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] =  cam_info['sensor2lidar_rotation']
            cam2lidar[:3, -1] = cam_info['sensor2lidar_translation']
            lidar2cam = np.linalg.inv(cam2lidar)

            # project the voxel to the pixel
            offsets= [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 1],
                      [0, 1, 0], [0, 1, 1], [1, 0, 0],
                      [1, 0, 1], [1, 1, 0], [1, 1, 1],
                      [0.5, 0.5, 0], [0.5, 0.5, 1],
                      [0.5, 0, 0.5], [0.5, 1, 0.5],
                      [0, 0.5, 0.5], [1, 0.5, 0.5],
                      ]
            for offset in offsets:
                offset_x, offset_y, offset_z = offset
                occ_full_index = np.stack(np.meshgrid(temp_z, temp_y, temp_x, indexing='ij'), axis=-1, ).reshape(-1, 3)  # in order: zyx
                occ_full_x = (occ_full_index[:, 2:3] + offset_x) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
                occ_full_y = (occ_full_index[:, 1:2] + offset_y) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
                occ_full_z = (occ_full_index[:, 0:1] + offset_z) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
                occ_full_points = np.concatenate((occ_full_x, occ_full_y, occ_full_z), axis=-1)
                occ_full_points = padding_column(occ_full_points)  # (n, 4)

                occ_index = np.arange(occ_full_points.shape[0])

                # map occ to pixels
                occ_points_in_cam = (lidar2cam @ (occ_full_points.T))[:-1]  # (3, n)
                occ_pixels_z = occ_points_in_cam[2, :]   # the depth of points
    
                occ_pixels = (cam_intrinsic @ occ_points_in_cam).T  # (n, 3)
                occ_pixels = occ_pixels[:, :2]/occ_pixels[:, 2:3]  # (n, 2)
                occ_pixels = np.around(occ_pixels).astype(np.int32)

                # the voxel in the fov of current camera including surface and occluded voxels
                fov_mask = np.logical_and(occ_pixels[:, 0] >= 0,
                                        np.logical_and(occ_pixels[:, 0] < self.img_w,
                                        np.logical_and(occ_pixels[:, 1] >= 0,
                                        np.logical_and(occ_pixels[:, 1] < self.img_h,
                                        occ_pixels_z > 0))))

                occ_pixels_fov = occ_pixels[fov_mask]  # (m, 2)
                occ_pixels_z_fov = occ_pixels_z[fov_mask]
                occ_gt_pixels_dict = cam_occ_gt_pixels_dict[cam_type]
                occ_index_fov = occ_index[fov_mask]

                invalid_occ_index = set()

                for i in range(len(occ_index_fov)):
                    cur_depth = occ_pixels_z_fov[i]
                    cur_pixel = (occ_pixels_fov[i][0], occ_pixels_fov[i][1])
                    # neighbors = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
                    neighbors = [(0, 0)]
                    for neighbor in neighbors:
                        cur_pixel = (cur_pixel[0]+neighbor[0], cur_pixel[1]+neighbor[1])
                        if cur_pixel in occ_gt_pixels_dict:
                            pixel_info = occ_gt_pixels_dict[cur_pixel]
                            if pixel_info['occ_depth'] < cur_depth and pixel_info['occ_label'] in self.background_labels:
                                invalid_occ_index.add(occ_index_fov[i])

                invalid_occ_indexs = invalid_occ_indexs | invalid_occ_index

        occ_gt_index = set(occ_gt_index)
        invalid_occ_indexs = invalid_occ_indexs - occ_gt_index

        invalid_occ_indexs = np.array(list(invalid_occ_indexs))

        return invalid_occ_indexs

    def split_occ_gt_into_surface_occluded(self, occ_gt, info):
        # occ GT points
        occ_gt_index = occ_gt[:, 0]
        occ_gt_label = occ_gt[:, 1]

        occ_gt_index_surfaces = set()
        cam_occ_gt_pixels_dict = {}

        for cam_type in info['cams']:  # 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            # if cam_type != 'CAM_FRONT_LEFT':
            #     continue
            cam_info = info['cams'][cam_type]
            cam_intrinsic = cam_info['cam_intrinsic']
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] =  cam_info['sensor2lidar_rotation']
            cam2lidar[:3, -1] = cam_info['sensor2lidar_translation']
            lidar2cam = np.linalg.inv(cam2lidar)

            # project the voxel to the pixel
            offsets= [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 1],
                      [0, 1, 0], [0, 1, 1], [1, 0, 0],
                      [1, 0, 1], [1, 1, 0], [1, 1, 1],
                      [0.5, 0.5, 0], [0.5, 0.5, 1],
                      [0.5, 0, 0.5], [0.5, 1, 0.5],
                      [0, 0.5, 0.5], [1, 0.5, 0.5],
                      ]

            pixel_dict_far = defaultdict(dict)  # 存储每个pixel ray对应的最远的voxel

            for offset in offsets:
                occ_gt_points = []
                offset_x, offset_y, offset_z = offset
                for i in range(len(occ_gt_index)):
                    indice = occ_gt_index[i]
                    x = indice % self.occ_xdim
                    y = (indice // self.occ_xdim) % self.occ_xdim
                    z = indice // (self.occ_xdim*self.occ_xdim)
                    point_x = (x + offset_x) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
                    point_y = (y + offset_y) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
                    point_z = (z + offset_z) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
                    occ_gt_points.append([point_x, point_y, point_z])  # order: x-y-z
                
                occ_gt_points = np.array(occ_gt_points)
                occ_gt_points = padding_column(occ_gt_points)  # (n, 4)
                    
                # occ gt 
                # map point from lidar to points
                occ_gt_points_in_cam = (lidar2cam @ (occ_gt_points.T))[:-1]  # (3, n)
                occ_gt_pixels_z = occ_gt_points_in_cam[2, :]   # the depth of points
    
                occ_gt_pixels = (cam_intrinsic @ occ_gt_points_in_cam).T  # (n, 3)
                occ_gt_pixels = occ_gt_pixels[:, :2]/occ_gt_pixels[:, 2:3]  # (n, 2)
                occ_gt_pixels = np.around(occ_gt_pixels).astype(np.int32)

                # the voxel in the fov of current camera including surface and occluded voxels
                fov_mask = np.logical_and(occ_gt_pixels[:, 0] >= 0,
                                        np.logical_and(occ_gt_pixels[:, 0] < self.img_w,
                                        np.logical_and(occ_gt_pixels[:, 1] >= 0,
                                        np.logical_and(occ_gt_pixels[:, 1] < self.img_h,
                                        occ_gt_pixels_z > 0))))
                
                occ_gt_pixels_fov = occ_gt_pixels[fov_mask]  # (m, 2)
                occ_gt_pixels_z_fov = occ_gt_pixels_z[fov_mask]
                
                occ_gt_index_fov = occ_gt_index[fov_mask]
                occ_gt_label_fov = occ_gt_label[fov_mask]

                pixel_dict_surface = defaultdict(dict)  
                for i in range(len(occ_gt_pixels_z_fov)):
                    u, v = occ_gt_pixels_fov[i][0], occ_gt_pixels_fov[i][1]
                    occ_depth = occ_gt_pixels_z_fov[i]
                    occ_index = occ_gt_index_fov[i] 
                    occ_label = occ_gt_label_fov[i]
                    if (u, v) not in pixel_dict_surface or occ_depth < pixel_dict_surface[(u, v)]['occ_depth']:
                        pixel_dict_surface[(u, v)]['occ_depth'] = occ_depth
                        pixel_dict_surface[(u, v)]['occ_index'] = occ_index
                        pixel_dict_surface[(u, v)]['occ_label'] = occ_label

                    if (u, v) not in pixel_dict_far or occ_depth > pixel_dict_far[(u, v)]['occ_depth']:
                        pixel_dict_far[(u, v)]['occ_depth'] = occ_depth
                        pixel_dict_far[(u, v)]['occ_index'] = occ_index
                        pixel_dict_far[(u, v)]['occ_label'] = occ_label
                
                unique_occ_gt_pixels = set(pixel_dict_surface.keys())
                occ_gt_index_surface = set()
                for key in pixel_dict_surface:
                    occ_gt_index_surface.add(pixel_dict_surface[key]['occ_index'])
    
                assert len(unique_occ_gt_pixels) == len(occ_gt_index_surface)
                occ_gt_index_surfaces = occ_gt_index_surfaces | occ_gt_index_surface
                # print(len(pixel_dict_far))
            cam_occ_gt_pixels_dict[cam_type] = pixel_dict_far

        return cam_occ_gt_pixels_dict, occ_gt_index_surfaces
    
    def obtain_occ_gt(self, points, points_label, points_velocity):    
        pc_range = np.array(self.point_cloud_range)
        voxel_size = np.array(self.occupancy_size)

        keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
                (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        points = points[keep, :]    
        points_label = points_label[keep]
        points_velocity = points_velocity[keep]
        # coords in the order: z-y-x 
        coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) / voxel_size[[2, 1, 0]]).astype(np.int32)
        # clip the coords
        z_coords, y_coords, x_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        z_coords = np.clip(z_coords, 0, self.occ_zdim-1)
        y_coords = np.clip(y_coords, 0, self.occ_ydim-1)
        x_coords = np.clip(x_coords, 0, self.occ_xdim-1)
        coords = np.stack((z_coords, y_coords, x_coords), axis=-1)
        voxel_dict = {}

        for i in range(len(points)):
            coords_index = tuple(coords[i])
            if coords_index not in voxel_dict:
                voxel_dict[coords_index] = {}
                voxel_dict[coords_index]['labels'] = [points_label[i]]
                voxel_dict[coords_index]['velocitys'] = [points_velocity[i]]
                voxel_dict[coords_index]['label_velocity_pair'] = defaultdict(list)
                voxel_dict[coords_index]['label_velocity_pair'][points_label[i]].append(points_velocity[i])
            else:
                voxel_dict[coords_index]['labels'].append(points_label[i])
                voxel_dict[coords_index]['velocitys'].append(points_velocity[i])
                voxel_dict[coords_index]['label_velocity_pair'][points_label[i]].append(points_velocity[i])
        
        occ_indexes = []
        occ_classes = []
        occ_velocitys = []
        skip_num = 0
        for index in voxel_dict:  # index is a tuple (index_z, index_y, index_x)
            point_num = len(voxel_dict[index]['labels'])
            voxel_label = np.argmax(np.bincount(voxel_dict[index]['labels']))
            # skip the background voxels with only one point
            if point_num <= self.voxel_point_threshold and voxel_label == self.occupancy_classes-1:
                skip_num = skip_num+1
                continue
            point_velocitys = voxel_dict[index]['label_velocity_pair'][voxel_label]
            point_velocitys = np.array(point_velocitys)
            voxel_velocity = np.mean(point_velocitys, axis=0)

            occ_index = index[0]*self.occ_xdim*self.occ_ydim + index[1]*self.occ_xdim + index[2] 
            occ_indexes.append(occ_index)
            occ_classes.append(voxel_label)
            occ_velocitys.append(voxel_velocity)

        occ_indexes =np.array(occ_indexes, dtype=np.int32).reshape(-1, 1)
        occ_classes = np.array(occ_classes, dtype=np.int32).reshape(-1, 1)

        occ_index_label = np.concatenate([occ_indexes, occ_classes], axis=-1)  # np.int32
        occ_velocitys = np.array(occ_velocitys)  # np.float32
        # print('skip num with point threshold:', skip_num)
        assert occ_index_label.shape[0] == occ_velocitys.shape[0]

        return occ_index_label, occ_velocitys
    
    def is_isolated_voxel_in_space(self, occ_index_sets, index):
        neighbors = [[-1, 0, 0], [1, 0, 0],
                     [0, -1, 0], [0, 1, 0],
                     [0, 0, -1], [0, 0, 1],
                    ]
        for neighbor in neighbors:
            neighbor_index = (index[0]+neighbor[0], index[1]+neighbor[1], index[2]+neighbor[2])
            neighbor_index_1d = neighbor_index[0]*self.occ_xdim*self.occ_ydim + \
                                neighbor_index[1]*self.occ_xdim + neighbor_index[2]
            if neighbor_index_1d in occ_index_sets:
                return False
        return True

    def is_isolated_voxel_on_road(self, occ_index_sets, index):
        neighbors = [[0, -1, 0], [0, 1, 0],
                     [0, 0, -1], [0, 0, 1],
                    ]
        for neighbor in neighbors:
            neighbor_index = (index[0]+neighbor[0], index[1]+neighbor[1], index[2]+neighbor[2])
            neighbor_index_1d = neighbor_index[0]*self.occ_xdim*self.occ_ydim + \
                                neighbor_index[1]*self.occ_xdim + neighbor_index[2]
            if neighbor_index_1d in occ_index_sets:
                return False
        return True
    
    def remove_isolated_voxel(self, occ_index_label, occ_velocitys):
        keep_mask = np.ones(occ_index_label.shape[0], dtype=bool)
        occ_index_sets = set(occ_index_label[:, 0])
        for i in range(occ_index_label.shape[0]):
            index, label = occ_index_label[i][0], occ_index_label[i][1]
            index_z = index//(self.occ_xdim*self.occ_ydim)
            index_y = (index - index_z*self.occ_xdim*self.occ_ydim)//self.occ_xdim
            index_x = index % self.occ_xdim
            index = (index_z, index_y, index_x)
            if self.is_isolated_voxel_in_space(occ_index_sets, index) and label == self.occupancy_classes-1:
                keep_mask[i] = False
            elif self.is_isolated_voxel_on_road(occ_index_sets, index) and label == self.occupancy_classes-1:
                keep_mask[i] = False
        print('remove isolated voxel num:', np.count_nonzero(keep_mask==False))
        # update the occ gt
        occ_index_label = occ_index_label[keep_mask]
        occ_velocitys = occ_velocitys[keep_mask]
        return occ_index_label, occ_velocitys


    def get_background_segment(self, occ_index_sets, occ_index_label, map_dict, index, label):
        neighbors = [[-1, 0, 0], [1, 0, 0],
                     [0, -1, 0], [0, 1, 0],
                     [0, 0, -1], [0, 0, 1],
                    ]
        segment_threshold = 10
        queues = [index]  # the index is an integer 
        heights = []
        count = 0
        collects = []
        visited = set()
        visited.add(index)
        while queues:
            cur_index = queues.pop(0)
            collects.append(cur_index)
            count = count + 1
            index_z = cur_index//(self.occ_xdim*self.occ_ydim)
            index_y = (cur_index - index_z*self.occ_xdim*self.occ_ydim)//self.occ_xdim
            index_x = cur_index % self.occ_xdim
            index_3d = (index_z, index_y, index_x)
            height = (index_z + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
            heights.append(height)

            if count > segment_threshold:  
                return False, None
            
            # BFS
            for neighbor in neighbors:
                new_index_3d = (index_3d[0]+neighbor[0], index_3d[1]+neighbor[1], index_3d[2]+neighbor[2])
                new_index_1d = new_index_3d[0]*self.occ_xdim*self.occ_ydim + \
                               new_index_3d[1]*self.occ_xdim + new_index_3d[2] 
                if new_index_1d in occ_index_sets:
                    new_label = occ_index_label[map_dict[new_index_1d]][1]
                    if new_index_1d not in visited and new_label == label:
                        queues.append(new_index_1d)
                        visited.add(new_index_1d)

        height_mean = np.mean(heights)
        if height_mean < -1: 
            return False, None
        else:
            return True, collects
        
    
    def remove_segments_in_air(self, occ_index_label, occ_velocitys):
        """
        remove_segments_in_air based on BFS
        """
        occ_index_sets = set(occ_index_label[:, 0])
        map_dict = {}   # occ_index_label[i][0] -> i 
        for ii in range(occ_index_label.shape[0]):
            index = occ_index_label[ii][0]
            map_dict[index] = ii
        
        removed_index = set()
        for i in range(occ_index_label.shape[0]):
            index, label = occ_index_label[i][0], occ_index_label[i][1]
            if label == self.occupancy_classes-1 and index not in removed_index:
                flag, local_sets = self.get_background_segment(occ_index_sets, occ_index_label, 
                                                               map_dict, index, label)
                if flag:
                    removed_index = removed_index.union(set(local_sets))
        print('the voxel num in the air:', len(removed_index))
        keep_mask = np.ones(occ_index_label.shape[0], dtype=bool)
        for index in removed_index:
            keep_mask[map_dict[index]] = False

        occ_index_label = occ_index_label[keep_mask]
        occ_velocitys = occ_velocitys[keep_mask]
        return occ_index_label, occ_velocitys


    def fill_empty(self, occ_index_sets, index3d, occ_index_label, occ_velocitys, map_dict):
    
        neighbors = [[0, -1, 0], [0, 1, 0],
                     [0, 0, -1], [0, 0, 1],
                     [0, -1, 1], [0, 1, 1],
                     [0, -1, -1], [0, 1, -1]
                    ]
        threshold = 6
        labels = []
        velocitys = []
        for neighbor in neighbors:
            neighbor_index_3d = (index3d[0]+neighbor[0], index3d[1]+neighbor[1], index3d[2]+neighbor[2])
            neighbor_index_1d = neighbor_index_3d[0]*self.occ_xdim*self.occ_ydim + \
                                neighbor_index_3d[1]*self.occ_xdim + neighbor_index_3d[2]
            if neighbor_index_1d in occ_index_sets:
                labels.append(occ_index_label[map_dict[neighbor_index_1d]][1])
                velocitys.append(occ_velocitys[map_dict[neighbor_index_1d]])
        
        if len(labels) >= threshold:
            vote_label = np.argmax(np.bincount(labels))  
            index = labels.index(vote_label)
            vote_velocity = velocitys[index]
            return True, vote_label, vote_velocity

        return False, None, None
        
    def fill_empty_voxel(self, occ_index_label, occ_velocitys):
        search_times = 5   
        for times in range(search_times):
            occ_index_sets = set(occ_index_label[:, 0])
            map_dict = {}   # occ_index_label[i][0] -> i 
            for ii in range(occ_index_label.shape[0]):
                index = occ_index_label[ii][0]
                map_dict[index] = ii

            new_occ_index_labels = []
            new_occ_velocitys = []
            for k in range(self.occ_zdim):
                for j in range(self.occ_ydim):
                    for i in range(self.occ_xdim):
                        index3d = (k, j, i)
                        index_1d = self.index3d_2_index1d(index3d)
                        if index_1d not in occ_index_sets:
                            valid, label, velocity = self.fill_empty(occ_index_sets, index3d, 
                                                                     occ_index_label, occ_velocitys, map_dict)
                            if valid:
                                new_occ_index_labels.append([index_1d, label])
                                new_occ_velocitys.append(velocity)
            new_occ_index_labels = np.array(new_occ_index_labels).astype(np.int32)
            new_occ_velocitys = np.array(new_occ_velocitys).astype(np.float32)
            if new_occ_index_labels.shape[0] != 0:
                occ_index_label = np.concatenate([occ_index_label, new_occ_index_labels], axis=0)
                occ_velocitys = np.concatenate([occ_velocitys, new_occ_velocitys], axis=0)
            # print('times:', times, 'new voxel number:', new_occ_index_labels.shape[0])

        return occ_index_label, occ_velocitys

    def obtain_points_in_box(self, points, box_gt):
        xc, yc, zc, length, width, height, theta = box_gt
        # pose: box2lidar   point in box-system to point in lidar-sytem
        box2lidar = np.eye(4)  
        box2lidar[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta),  0],
                                      [0,             0,              1]])
        box2lidar[:3, -1] = np.array([xc, yc, zc])

        points_pad = padding_column(points)  # (n, 4)

        lidar2box = np.linalg.inv(box2lidar)  # point in lidar-system to point in box-system
        point_in_box_sys = np.dot(lidar2box, points_pad.T).T  # (n ,4)

        mask = np.ones(points_pad.shape[0], dtype=bool)
        ratio = 1.0  # enlarge the box to ensure the quality of background points
        mask = np.logical_and(mask, abs(point_in_box_sys[:, 0]) <= length/2*ratio)
        mask = np.logical_and(mask, abs(point_in_box_sys[:, 1]) <= width/2*ratio)
        mask = np.logical_and(mask, abs(point_in_box_sys[:, 2]) <= height/2*ratio)

        return mask

    def split_lidar_with_bbox(self, points):
        """restrict the use in the real application"""
        info = self.info
        anns_results = self.get_ann_info()
        gt_bboxes_3d = anns_results['gt_bboxes_3d']  # (x, y, z_bottom, w, l, h, theta)
        gt_bboxes_3d = gt_bboxes_3d.tensor.numpy()
        gt_labels_3d = anns_results['gt_labels_3d']
        gt_names_3d = anns_results['gt_names']
        assert gt_bboxes_3d.shape[-1] == 9
        gt_bboxes_velocity = gt_bboxes_3d[:, -2:]
        points_label = np.ones(points.shape[0], dtype=np.int8)*self.occupancy_classes  # background label先归为一类，后面细分
        points_velocity = np.zeros((points.shape[0], 2), dtype=np.float32)

        if len(gt_labels_3d) != 0:
            # change to 3d_box_st [x, y, z, l, w, h, theta_counterclockwise]
            
            _gt_bboxes_3d = gt_bboxes_3d.copy()[:, :7]
            _gt_bboxes_3d[:, 2] += _gt_bboxes_3d[:, 5]*0.5
            _gt_bboxes_3d[:, 3:6] += self.enlarge_box
            _gt_bboxes_3d = _gt_bboxes_3d[:, [0, 1, 2, 4, 3, 5, 6]]
            _gt_bboxes_3d[:, -1] = -(_gt_bboxes_3d[:, -1]+np.pi/2)

            for i in range(len(gt_labels_3d)):
                if gt_labels_3d[i] >= 0:  # skip non-used class
                    in_box_mask = self.obtain_points_in_box(points, _gt_bboxes_3d[i])
                    points_label[in_box_mask] = gt_labels_3d[i] 
                    points_velocity[in_box_mask] = gt_bboxes_velocity[i]

        return points, points_label, points_velocity  # (n, 3), (n), (n, 2)

    def get_ann_info(self):
        info = self.info
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)  # 7+2=9

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5))

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results







