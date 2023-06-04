# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
"""
convert nuscenes and occupnacy data to generate pkls scene by scene
accumulate the lidar data: sample+sweep
(1) accumulate the background and foreground objects separately
(2) introduce the lidarseg in sample data
(3) generate the occupancy gt
"""
import enum
from itertools import accumulate
import shutil
import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

from utils import *
import cv2
import json
import os.path as osp
import yaml
from single_frame_occ_generator import SingleFrameOCCGTGenerator

def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          occ_resolution='normal',
                          save_flow_info=True):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, out_path, test, max_sweeps=max_sweeps,
        occ_resolution=occ_resolution, save_flow_info=save_flow_info)

def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)

def split_scenes(scenes):
    scenes = sorted(scenes)
    num_tasks = int(os.environ['SLURM_NTASKS'])
    cur_id = int(os.environ['SLURM_PROCID'])
    # train_scenes
    num_scene = len(scenes)
    a = num_scene//num_tasks
    b = num_scene % num_tasks

    if cur_id == 0:
        print('num_scene:', num_scene)

    process_num = []
    count = 0
    for id in range(num_tasks):
        if id >= b:
            process_num.append(a)
        else:
            process_num.append(a+1)
    addsum = np.cumsum(process_num)

    if cur_id == 0:
        start = 0
        end = addsum[0]
    else:
        start = addsum[cur_id-1]
        end = addsum[cur_id]

    return scenes[start:end]


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         out_dir,
                         test=False,
                         max_sweeps=10,
                         occ_resolution='normal',
                         save_flow_info=True):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # split the train_scenes and val_scenes
    train_scenes = split_scenes(train_scenes)
    val_scenes = split_scenes(val_scenes)

    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    # define a scene dict
    scene_dict = {}
    save_surround_images = True
    save_scene_background_point = False
    save_instance_point = False 
    save_single_frame_data = False  # save accumulated lidar frame-by-frame


    for sample in mmcv.track_iter_progress(nusc.sample):
        cur_frame_idx = frame_idx
        scene_token = sample['scene_token']
        if frame_idx == 0:
            scene_dict[scene_token] = {}

        if (scene_token not in train_scenes) and (scene_token not in val_scenes):
            continue

        train_flag = False
        if sample['scene_token'] in train_scenes:
            train_flag = True

        scene = nusc.get('scene', sample['scene_token'])
        scene_name = scene['name']
        
        # if scene_name != 'scene-0001':
        #     continue

        if train_flag:
            scene_save_dir = os.path.join(out_dir, 'train', scene_name)
        else:
            scene_save_dir = os.path.join(out_dir, 'val', scene_name)
        os.makedirs(scene_save_dir, exist_ok=True)

        scene_pkl_file = os.path.join(scene_save_dir, 'scene_info.pkl')
        if os.path.exists(scene_pkl_file):  # this scene has been processed
            continue

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        lidar_seg_path = osp.join(nusc.dataroot, nusc.get('lidarseg', sample['data']['LIDAR_TOP'])['filename'])

        mmcv.check_file_exist(lidar_path)
        mmcv.check_file_exist(lidar_seg_path)

        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        ##
        info = {
            'lidar_path': lidar_path,
            'lidar_seg_path': lidar_seg_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'scene_name': scene_name,  # additional info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # add lidar2global: map point coord in lidar to point coord in the global
        l2e = np.eye(4)
        l2e[:3, :3] = l2e_r_mat
        l2e[:3, -1] = l2e_t
        e2g = np.eye(4)
        e2g[:3, :3] = e2g_r_mat
        e2g[:3, -1] = e2g_t
        lidar2global  = np.dot(e2g, l2e)
        info['ego2global'] = e2g
        info['lidar2ego'] = l2e
        info['lidar2global'] = lidar2global  # additional info

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                temp = nusc.get('sample_data', sweep['sample_data_token'])
                if temp['is_key_frame']:   # TODO we only add the real sweep data (not key frame)
                    break
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])

            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            # get the box id for tracking the box in the scene
            instance_tokens = [item['instance_token'] for item in annotations]

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            dims_lwh = np.concatenate([dims[:, 1:2], dims[:, 0:1], dims[:, 2:]], axis=-1)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            velocity_3d = np.array([nusc.box_velocity(token) for token in sample['anns']])

            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar: only need the rotation matrix
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            for i in range(len(boxes)):
                velo = velocity_3d[i]
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity_3d[i] = velo

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            gt_boxes_st = np.concatenate([locs, dims_lwh, rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['gt_velocity_3d'] = velocity_3d.reshape(-1, 3)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

            # additional info
            info['instance_tokens'] = instance_tokens
            info['dims_lwh'] = dims_lwh
            info['gt_boxes_st'] = gt_boxes_st # standard definition of gt_boxes_st

            back_instance_info = extract_frame_background_instance_lidar(lidar_path, info)
            back_instance_info_sweeps, sweep_infos = extract_and_split_sweep_lidar(nusc, info)

            background_lidar_seg_info = extract_background_lidar_seg(lidar_path, lidar_seg_path, info)

            save_sample_lidar = False
            if save_sample_lidar:
                sample_lidar_dir = os.path.join(scene_save_dir, 'samples')
                os.makedirs(sample_lidar_dir, exist_ok=True)
                sample_save_path = os.path.join(sample_lidar_dir, '{:03}.bin'.format(cur_frame_idx))
                shutil.copy(info['lidar_path'], sample_save_path)

            check_sweep_data = False
            if check_sweep_data:  # check lidar and 3D box
                sweep_save_dir = os.path.join(scene_save_dir, 'sweeps')
                os.makedirs(sweep_save_dir, exist_ok=True)
                for sweep_index, sweep_info in enumerate(sweep_infos):
                    sweep_lidar_path = sweep_info['lidar_path']
                    sweep_save_path = os.path.join(sweep_save_dir, '{:03}_{}_.bin'.format(cur_frame_idx, sweep_index))
                    shutil.copy(sweep_lidar_path, sweep_save_path)
                    result_box = {'gt_bboxes_3d': sweep_info['gt_boxes_st'],
                                  'gt_labels_3d': sweep_info['gt_names']}
                    box_save_path = os.path.join(sweep_save_dir, '{:03}_{}_box.pkl'.format(cur_frame_idx, sweep_index))
                    mmcv.dump(result_box, box_save_path)

            scene_dict[scene_token][cur_frame_idx] = {}
            scene_dict[scene_token][cur_frame_idx]['basic_info'] = info
            scene_dict[scene_token][cur_frame_idx]['back_instance_info'] = back_instance_info
            scene_dict[scene_token][cur_frame_idx]['back_instance_info_sweeps'] = back_instance_info_sweeps
            scene_dict[scene_token][cur_frame_idx]['background_lidar_seg_info'] = background_lidar_seg_info

            accum_sweep = True
            if frame_idx == 0:  # end of the current scene
                background_track, instance_track = accumulate_background_box_point(scene_dict[scene_token], accum_sweep)

                lidarseg_background_points, lidarseg_background_labels = accumulate_lidarseg_background(scene_dict[scene_token])


                # 1. save wholce scene accumulated background and box points separately
                # save whole scene accumulated background points in the global system
                back_points = background_track['accu_global']
                if save_scene_background_point:
                    scene_background_path = os.path.join(scene_save_dir,'scene_background_point.bin')
                    back_points_save = back_points.astype(np.float32)
                    back_points_save.tofile(scene_background_path)  

                # save the accumulated box points in the local box system
                if save_instance_point:
                    instance_points_dir = os.path.join(scene_save_dir, 'instance_points')
                    os.makedirs(instance_points_dir, exist_ok=True)
                    for instance_id in instance_track:
                        instance_object_track = instance_track[instance_id]
                        box_points = instance_object_track.accu_points
                        box_points = box_points.astype(np.float32)
                        class_name = instance_object_track.class_name
                        class_dir = os.path.join(instance_points_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)

                        if instance_object_track.is_stationary:
                            box_points.tofile(os.path.join(class_dir,instance_id+'_sta.bin'))
                        else:
                            box_points.tofile(os.path.join(class_dir,instance_id+'_mov.bin'))

                # 2. save the accu points  and gt box in the lidar system frame by frame
                save_frame_idxs = sorted(scene_dict[scene_token].keys())
                
                if save_single_frame_data:
                    single_frame_dir = os.path.join(scene_save_dir, 'sing_frame_data')
                    os.makedirs(single_frame_dir, exist_ok=True)
                cur_scene_infos = []
                for cur_frame_idx in save_frame_idxs:
                    # save accumulate sample lidar data
                    cur_info = scene_dict[scene_token][cur_frame_idx]['basic_info']
                    lidar2global =cur_info['lidar2global']

                    back_points_in_lidar = transform_points_global2lidar(back_points, lidar2global, filter=True)
                    flag, box_points_in_lidar, points_velocitys = transform_points_box2lidar(instance_track, cur_frame_idx)
                    if flag:
                        points_in_lidar = np.concatenate([box_points_in_lidar, back_points_in_lidar], axis=0)
                    else:
                        points_in_lidar = back_points_in_lidar
                    points_in_lidar = points_in_lidar.astype(np.float32)

                    if save_single_frame_data:
                        cur_frame_save_path = os.path.join(single_frame_dir, '{:03d}.bin'.format(cur_frame_idx))
                        points_in_lidar.tofile(cur_frame_save_path)
                        cur_info['accumulate_lidar_path'] = cur_frame_save_path


                    # save sample data lidarseg
                    lidarseg_background_points_frame, lidarseg_background_labels_frame = transform_lidarseg_back_global2lidar(
                        lidarseg_background_points,
                        lidarseg_background_labels,
                        lidar2global,
                        filter=True)
                    lidarseg_background_points_frame = lidarseg_background_points_frame.astype(np.float32)
                    assert len(lidarseg_background_points_frame) == len(lidarseg_background_labels_frame)
                    
                    if save_single_frame_data:
                        lidarseg_point_path = os.path.join(single_frame_dir, '{:03d}_lidarseg_back_points.bin'.format(cur_frame_idx))
                        lidarseg_background_points_frame.tofile(lidarseg_point_path)
                        cur_info['lidarseg_back_points_path'] = lidarseg_point_path
                        lidarseg_label_path = os.path.join(single_frame_dir, '{:03d}_lidarseg_back_labels.npy'.format(cur_frame_idx))
                        np.save(lidarseg_label_path, lidarseg_background_labels_frame)
                        cur_info['lidarseg_label_path'] = lidarseg_label_path

                    # save the gt box
                    if save_single_frame_data:
                        result_box = {'gt_bboxes_3d': cur_info['gt_boxes_st'],
                                    'gt_labels_3d': cur_info['gt_names'],
                                    'gt_velocity': cur_info['gt_velocity']}
                        box_save_path = os.path.join(single_frame_dir, '{:03}_box.pkl'.format(cur_frame_idx))
                        mmcv.dump(result_box, box_save_path)

                    # save surround images for visualization
                    if save_surround_images:
                        img_save_dir = os.path.join(scene_save_dir, 'surround_images')
                        os.makedirs(img_save_dir, exist_ok=True)

                        cams = cur_info['cams']
                        cam_types = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
                        height, width = 900, 1600
                        images = []
                        for cam_type in cam_types:
                            data_path = cams[cam_type]['data_path']
                            img = cv2.imread(data_path)
                            images.append(img)
                        six_image = np.zeros((height*2, width*3, 3))
                        six_image[:height, :width] = images[0]
                        six_image[:height, width:width*2] = images[1]
                        six_image[:height, width*2:width*3] = images[2]
                        six_image[height:2*height, :width] = images[3]
                        six_image[height:2*height, width:width*2] = images[4]
                        six_image[height:2*height, width*2:width*3] = images[5]
                        surround_image_path = os.path.join(img_save_dir, '{:03d}.jpg'.format(cur_frame_idx))
                        cv2.imwrite(surround_image_path, six_image)
                        cur_info['surround_image_path'] = surround_image_path

                    generator = SingleFrameOCCGTGenerator(cur_info, scene_save_dir, train_flag=train_flag,
                                                          occ_resolution=occ_resolution, voxel_point_threshold=0,
                                                          save_flow_info=save_flow_info)

                    occ_paths = generator.save_occ_gt(points_in_lidar,
                                          lidarseg_background_points_frame, 
                                          lidarseg_background_labels_frame)
                    
                    occ_gt_path, flow_gt_path, occ_invalid_path = occ_paths
                    cur_info['occ_gt_path'] = occ_gt_path
                    cur_info['flow_gt_path'] = flow_gt_path
                    cur_info['occ_invalid_path'] = occ_invalid_path

                    # save scene info
                    cur_scene_infos.append(cur_info)

                mmcv.dump(cur_scene_infos, scene_pkl_file)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def merge_scene(root_dir):
    for data_type in ['train', 'val']:
        print('process scenes:', data_type)
        scenes_dir = os.path.join(root_dir, data_type)
        if not os.path.exists(scenes_dir):
            continue
        datas = []
        for scene in sorted(os.listdir(scenes_dir)):
            data = mmcv.load(os.path.join(scenes_dir, scene, 'scene_info.pkl'))
            datas.extend(data)
        if data_type == 'train':
            assert len(datas) == 28130
        else:
            assert len(datas) == 6019
        save_path = os.path.join(root_dir, 'nuscenes_infos_temporal_{}.pkl'.format(data_type))
        metadata = dict(version='v1.0-trainval')
        save_data = dict(infos=datas, metadata=metadata)
        mmcv.dump(save_data, save_path)

if __name__ == '__main__':
    root_path = './data/nuscenes'
    can_bus_root_path = './data'
    version = 'v1.0-trainval'
    info_prefix = 'nuscenes'

    occ_resolution = 'normal' 
    out_dir = './data/nuscenes_occupancy_gt_v1_0'

    save_flow_info=True
    create_nuscenes_infos(root_path, out_dir, can_bus_root_path, info_prefix,
                          version=version, occ_resolution=occ_resolution, save_flow_info=save_flow_info)

