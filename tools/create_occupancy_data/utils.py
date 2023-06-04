import numpy as np
from pyquaternion import Quaternion
from mmdet3d.datasets import NuScenesDataset
import yaml


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

yaml_path = 'tools/create_occupancy_data/nuscenes_lidar_class.yaml'
with open(yaml_path) as f:
    lidar_class_map = yaml.full_load(f)
lidar_class_map = lidar_class_map['learning_map']
                  
class InstanceObjectTrack():
    def __init__(self, instance_id, frame_id=0, instance_object=None):
        # self.scene_id = scene_id
        self.instance_id = instance_id
        self.track_dict = {} 
        self.track_dict[frame_id] = instance_object
        self.class_name = instance_object.class_name
        self.accu_points = instance_object.points
        self.start_position = instance_object.global_center
        self.end_position = instance_object.global_center
    
    def add(self, frame_id, instance_object):
        self.track_dict[frame_id] = instance_object
        self.end_position = instance_object.global_center


    @property
    def is_stationary(self):
        postion_diff = (self.end_position - self.start_position)[:2]
        thre = 1
        if np.linalg.norm(postion_diff) > thre:
            return False
        else:
            return True

    def __repr__(self):
        return 'track instance_id: {}'.format(self.instance_id)

class InstanceObject():
    """define the object instance"""
    def __init__(self, instance_id, class_name, gt_box, lidar2global, velocity=None):
        self.instance_id = instance_id
        self.class_name = class_name
        xc, yc, zc, length, width, height, theta = gt_box  # defined in the lidar coordinate
        self.length = length
        self.width = width
        self.height = height
        self.xc, self.yc, self.zc = xc, yc, zc 
        # pose: box2lidar
        self.pose = np.eye(4)  
        self.pose[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta),  0],
                                      [0,             0,              1]])
        self.pose[:3, -1] = np.array([xc, yc, zc])

        assert len(velocity) == 2
        if velocity is None:
            self.vel_x, self.vel_y = 0, 0
        elif np.isnan(velocity).any():
            self.vel_x, self.vel_y = 0, 0
        else:
            self.vel_x = velocity[0]
            self.vel_y = velocity[1]

        # find the global coordinate of the box center
        self.global_center = self.get_global_center(xc, yc, zc, lidar2global)
        self.points = None
    
    def get_global_center(self, xc, yc, zc, lidar2global):
        point = np.array([xc, yc, zc, 1]).reshape(4, 1)
        global_center = np.dot(lidar2global, point).squeeze()
        return global_center[:3]

    def __repr__(self):
        return 'track instance_id: {}'.format(self.instance_id)

def load_lidar(lidar_path, remove_close=False):
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points = points[:, :3]  # (n, 3)

    if remove_close:
        x_radius = 3.0
        y_radius = 3.0
        z_radius = 2.0
        x_filt = np.abs(points[:, 0]) < x_radius
        y_filt = np.abs(points[:, 1]) < y_radius
        z_filt = np.abs(points[:, 2]) < z_radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt, z_filt))
        return points[not_close]

    return points

def padding_column(points):
    # padding 1 in the last column: change the dimension of points from (n, 3) to (n, 4)
    ones = np.ones_like(points[:, 0]).reshape(-1, 1)
    homo_points = np.concatenate((points, ones), axis=-1)
    return homo_points

def split_points(background_points, instance_object):
    """
    Args:
        background_points: (n, 3)

    Returns:
        background_points: (n,3) background point defined in the lidar system
        points_in_box: (n, 3) in_box point defined in the object system
    """
   
    points = padding_column(background_points)  # (n, 4)

    box2lidar = instance_object.pose  # point in box-system to point in lidar-sytem
    lidar2box = np.linalg.inv(box2lidar)  # point in lidar-system to point in box-system
    point_in_box_sys = np.dot(lidar2box, points.T).T  # (n ,4)

    length, width, height = instance_object.length, instance_object.width, instance_object.height

    mask = np.ones(points.shape[0], dtype=bool)
    # ratio = 1.1  # enlarge the box to ensure the quality of background points
    # mask = np.logical_and(mask, abs(point_in_box_sys[:, 0]) < length/2*ratio)
    # mask = np.logical_and(mask, abs(point_in_box_sys[:, 1]) < width/2*ratio)
    # mask = np.logical_and(mask, abs(point_in_box_sys[:, 2]) < height/2*ratio)

    fill_scale = 0.25   # force fine occupancy
    mask = np.logical_and(mask, abs(point_in_box_sys[:, 0]) <= (length+fill_scale)/2)
    mask = np.logical_and(mask, abs(point_in_box_sys[:, 1]) <= (width+fill_scale)/2)
    mask = np.logical_and(mask, abs(point_in_box_sys[:, 2]) <= (height+fill_scale)/2)

    points_in_box = point_in_box_sys[mask][:, :3]
    background_points = background_points[mask==False]

    return background_points, points_in_box

def extract_frame_background_instance_lidar(lidar_path, info):
    """
    将每一帧点云分割为背景点云+instance点云
    背景点云: 保存在世界坐标系
    instance点云: 保存在物体坐标系中
    """
    background_points = load_lidar(lidar_path, remove_close=True)

    instance_tokens = info['instance_tokens']
    gt_boxes_st = info['gt_boxes_st'] 
    gt_names = info['gt_names']
    lidar2global = info['lidar2global']
    gt_velocitys = info['gt_velocity']
    back_instance_info = {}
    back_instance_info['instance'] = {}
    for i in range(len(instance_tokens)):
        instance_token = instance_tokens[i] 
        class_name = gt_names[i] 
        if class_name not in nus_categories:   # TODO only consider the nus class, other as the background
            continue
        gt_box = gt_boxes_st[i] 
        gt_velocity = gt_velocitys[i]
        instance_object = InstanceObject(instance_token, class_name, gt_box, lidar2global, velocity=gt_velocity)
        background_points, instance_points = split_points(background_points, instance_object)
        instance_object.points = instance_points
        back_instance_info['instance'][instance_token] = instance_object
    
    # transfer background points from lidar to global
    background_points = transform_points_lidar2global(background_points, lidar2global)
    back_instance_info['background_points'] = background_points

    return back_instance_info

def accumulate_box_point(instance_object_track):
    accu_points = []
    track_dict = instance_object_track.track_dict
    for frame_idx in track_dict:
        accu_points.append(track_dict[frame_idx].points)
    
    accu_points = np.concatenate(accu_points, axis=0)
    instance_object_track.accu_points = accu_points
    return 

def accumulate_background_point(background_track, scene_info, accum_sweep=False):
    accu_global_points = [] 
    for frame_idx in background_track.keys():
        # if frame_idx >10:
        #     continue
        if isinstance(frame_idx, int):
            points = background_track[frame_idx]
        if isinstance(frame_idx, str) and accum_sweep:
            points = background_track[frame_idx]
        accu_global_points.append(points)

    accu_global_points = np.concatenate(accu_global_points, axis=0)
    return accu_global_points

def accumulate_neighbor_background(background_track, cur_frame_idx, 
                                   accum_sweep=False, neighbor_frame=10):
    """
    only accumulate_neighbor_background, do not accumulate the whole scene
    """
    accu_global_points = [] 
    for frame_idx in background_track.keys():
        if isinstance(frame_idx, int):
            if abs(frame_idx - cur_frame_idx) <= neighbor_frame:
                points = background_track[frame_idx]
                accu_global_points.append(points)
        if isinstance(frame_idx, str) and accum_sweep:
            if len(frame_idx) > 5:  # useless data
                continue
            if abs(int(frame_idx.split('_')[0]) - cur_frame_idx) <= neighbor_frame:
                points = background_track[frame_idx]
                accu_global_points.append(points)

    accu_global_points = np.concatenate(accu_global_points, axis=0)
    return accu_global_points

def accumulate_background_box_point(scene_info, accum_sweep=False):
    """accumulate the point in a sequence with about 40 frames"""
    instance_track = {} # {instance_id: InstanceObjectTrack}
    background_track = {}  # {frame_id: points, 'acumulate': points}

    # 1. generate the object track and background track in the whole scene
    for frame_idx in sorted(scene_info.keys()):
        basic_info = scene_info[frame_idx]['basic_info']
        back_instance_info = scene_info[frame_idx]['back_instance_info']
        background_track[frame_idx] = back_instance_info['background_points']
        back_instance_info = scene_info[frame_idx]['back_instance_info']

        for instance_id in back_instance_info['instance']:
            instance_object = back_instance_info['instance'][instance_id]
            if instance_id not in instance_track:
                instance_track[instance_id] = InstanceObjectTrack(instance_id, frame_idx, instance_object)
            instance_track[instance_id].add(frame_idx, instance_object) 
        
        # add sweep data
        back_instance_info_sweeps = scene_info[frame_idx]['back_instance_info_sweeps'] # this is a list
        if accum_sweep:
            for index, info_sweep in enumerate(back_instance_info_sweeps):
                temp_frame_idx = str(frame_idx)+'_'+str(index)
                background_track[temp_frame_idx] = info_sweep['background_points']
                for instance_id in info_sweep['instance']:
                    instance_object = info_sweep['instance'][instance_id]
                    if instance_id not in instance_track:
                        instance_track[instance_id] = InstanceObjectTrack(instance_id, temp_frame_idx, instance_object)
                    instance_track[instance_id].add(temp_frame_idx, instance_object)
                

    # 2. accumulate object point cloud in the object system
    for instance_id in instance_track:
        instance_object_track = instance_track[instance_id]
        accumulate_box_point(instance_object_track)
        # print('shape:', instance_object_track.accu_points.shape)

    # 3. accumulate background points in the global system TODO 
    background_track['accu_global'] = accumulate_background_point(background_track, scene_info, accum_sweep)

    return background_track, instance_track

def transform_points_global2lidar(back_points, lidar2global, filter=False):
    back_points_homo = padding_column(back_points).T
    points = (np.linalg.inv(lidar2global) @ back_points_homo).T[:, :3]

    if filter:  # filter the point out of range
        # print('befor filter:', points.shape)  # 900w
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
                (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        points = points[keep, :]    
        # print('after filter:', points.shape)  # 500w

    return points

def transform_points_lidar2global(back_points, lidar2global):
    back_points_homo = padding_column(back_points).T  # (4, n)
    points_in_global = (lidar2global @ back_points_homo).T[:, :3]
    return points_in_global

def transform_points_box2lidar(instance_track, cur_frame_idx):
    points = []
    points_velocitys = []
    for instance_id in instance_track:
        track_dict = instance_track[instance_id].track_dict
        accu_points = instance_track[instance_id].accu_points
        if cur_frame_idx in track_dict:
            box2lidar = track_dict[cur_frame_idx].pose
            accu_points_homo = padding_column(accu_points).T
            points_in_lidar = (box2lidar @ accu_points_homo).T[:, :3]
            box_vel_x, box_vel_y = track_dict[cur_frame_idx].vel_x, track_dict[cur_frame_idx].vel_y
            points_num = points_in_lidar.shape[0]
            points_vel = np.tile(np.array([box_vel_x, box_vel_y]), (points_num, 1))
            points.append(points_in_lidar)
            points_velocitys.append(points_vel)
    if points == []:  # 当前帧可能没有标注的box
        points = np.array(points)
        points_velocitys = np.array(points_velocitys)
        return False, points, points_velocitys
    else:
        points = np.concatenate(points, axis=0)  # (n, 3)
        points_velocitys = np.concatenate(points_velocitys, axis=0)  # (n, 2)
        return True, points, points_velocitys


def calculate_lidar2global(l2e_r, l2e_t, e2g_r, e2g_t):
    """
    transfer point in lidar system to point in the global world system
    """
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
    return lidar2global
    
# process sweep data
def extract_and_split_sweep_lidar(nusc, info):
    "the raw data is defined in the sweep lidar system"
    sweeps = info['sweeps']
    back_instance_infos = []
    sweep_infos = []
    for sweep in sweeps:
        data_path = sweep['data_path']
        lidar_token = sweep['sample_data_token']
        l2e_r = sweep['sensor2ego_rotation']
        l2e_t = sweep['sensor2ego_translation']
        e2g_r = sweep['ego2global_rotation']
        e2g_t = sweep['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        lidar2global = calculate_lidar2global(l2e_r, l2e_t, e2g_r, e2g_t)

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        boxes_tokens = [box.token for box in boxes]
        annotations = [nusc.get('sample_annotation', token) for token in boxes_tokens]
        instance_tokens = [item['instance_token'] for item in annotations]

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        dims_lwh = np.concatenate([dims[:, 1:2], dims[:, 0:1], dims[:, 2:]], axis=-1)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                            for b in boxes]).reshape(-1, 1)
        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NuScenesDataset.NameMapping:
                names[i] = NuScenesDataset.NameMapping[names[i]]
        names = np.array(names)
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        gt_boxes_st = np.concatenate([locs, dims_lwh, rots], axis=1)


        velocity = np.array(
            [nusc.box_velocity(token)[:2] for token in boxes_tokens])
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity[i] = velo[:2]

        sweep_info = {
            'lidar_path': lidar_path,
            'instance_tokens': instance_tokens,
            'gt_boxes_st': gt_boxes_st,
            'gt_names': names,
            'lidar2global': lidar2global,
            'gt_velocity': velocity.reshape(-1, 2),
        }
        sweep_infos.append(sweep_info)
        back_instance_info = extract_frame_background_instance_lidar(lidar_path, sweep_info)
        back_instance_infos.append(back_instance_info)

    return  back_instance_infos, sweep_infos



"""
the following: extract lidar seg info
"""
def extract_background_lidar_seg(lidar_path, lidar_seg_path, info):
    # map seg label from 0-31 to 0-16 只考虑noise+6类背景 [0, 11, 12, 13, 14, 15, 16]
    lidar_seg_label = np.fromfile(lidar_seg_path, dtype=np.uint8)  # 0-31
    for i in range(len(lidar_seg_label)):
        lidar_seg_label[i] = lidar_class_map[lidar_seg_label[i]]  # 0-31 map to 0-16

    lidar_points = load_lidar(lidar_path)
    lidar2global = info['lidar2global']
    lidar_points = transform_points_lidar2global(lidar_points, lidar2global)
    background_mask = np.zeros_like(lidar_seg_label).astype(np.bool)
    background_labels = {0, 11, 12, 13, 14, 15, 16}
    for i in range(len(background_mask)):
        if lidar_seg_label[i] in background_labels:
            background_mask[i] = True

    background_lidar_seg_info = {}
    background_lidar_seg_info['points'] = lidar_points[background_mask]
    background_lidar_seg_info['labels'] = lidar_seg_label[background_mask]

    return background_lidar_seg_info

def accumulate_lidarseg_background(scene_info):
    """累积lidarseg背景点及对应的labels"""
    lidarseg_background_points_accum = []
    lidarseg_background_labels_accum = []
    for frame_idx in sorted(scene_info.keys()):
        background_lidar_seg_info = scene_info[frame_idx]['background_lidar_seg_info']
        lidar_points = background_lidar_seg_info['points']
        lidar_labels = background_lidar_seg_info['labels']

        lidarseg_background_points_accum.append(lidar_points)
        lidarseg_background_labels_accum.append(lidar_labels)

    lidarseg_background_points_accum = np.concatenate(lidarseg_background_points_accum, axis=0)
    lidarseg_background_labels_accum = np.concatenate(lidarseg_background_labels_accum, axis=0)

    return lidarseg_background_points_accum, lidarseg_background_labels_accum



def transform_lidarseg_back_global2lidar(points, labels, lidar2global, filter=False):
    points_homo = padding_column(points).T
    points = (np.linalg.inv(lidar2global) @ points_homo).T[:, :3]

    if filter:  # filter the point out of range
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
                (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        points = points[keep, :]    
        labels = labels[keep]

    return points, labels