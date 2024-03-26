import pickle
import numpy as np
from pyquaternion import Quaternion
import torch
from torch.utils.data import Dataset
np.set_printoptions(precision=3, suppress=True)

def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm

class EgoPoseDataset(Dataset):
    def __init__(self, data_infos, dataset_type=None):
        super(EgoPoseDataset, self).__init__()

        self.data_infos = data_infos
        assert dataset_type in ['openocc_v2', 'lightwheelocc']
        self.dataset_type = dataset_type

        if self.dataset_type == 'lightwheelocc':
            # lightwheelocc doesn't have lidar now, we use pseudo lidar2ego instead
            self.pseudo_lidar2ego = np.array([
                [ 0., 1., 0., 0.94 ],
                [-1., 0., 0., 0.   ],
                [ 0., 0., 1., 1.84 ],
                [ 0., 0., 0., 1.   ]])

        self.scene_frames = {}
        for info in data_infos:
            scene_token = self.get_scene_token(info)
            if scene_token not in self.scene_frames:
                self.scene_frames[scene_token] = []
            self.scene_frames[scene_token].append(info)

    def __len__(self):
        return len(self.data_infos)

    def get_scene_token(self, info):
        if self.dataset_type == 'openocc_v2':
            # meta info of openocc_v2 don't have scene_token
            # extract scene name from 'occ_path' instead
            # if the custom data info contains scene_token, we just use it.
            if 'scene_token' in info:
                scene_name = info['scene_token']
            else:
                scene_name = info['occ_path'].split('openocc_v2/')[-1].split('/')[0]
            return scene_name
        elif self.dataset_type == 'lightwheelocc':
            return info['scene_token']
        else:
            raise ValueError('Invalid dataset type')
        
    def get_ego_from_lidar(self, info):
        if self.dataset_type == 'openocc_v2':
            ego_from_lidar = trans_matrix(
                np.array(info['lidar2ego_translation']), 
                Quaternion(info['lidar2ego_rotation']))
        elif self.dataset_type == 'lightwheelocc':
            # lightwheelocc doesn't have lidar2ego, use pseudo lidar2ego instead
            ego_from_lidar = self.pseudo_lidar2ego
        return ego_from_lidar

    def get_global_pose(self, info, inverse=False):

        global_from_ego = trans_matrix(
            np.array(info['ego2global_translation']), 
            Quaternion(info['ego2global_rotation']))
        if self.dataset_type == 'openocc_v2':
            ego_from_lidar = trans_matrix(
                np.array(info['lidar2ego_translation']), 
                Quaternion(info['lidar2ego_rotation']))
        elif self.dataset_type == 'lightwheelocc':
            # lightwheelocc doesn't have lidar2ego, use pseudo lidar2ego instead
            ego_from_lidar = self.pseudo_lidar2ego

        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        ref_sample_token = info['token']
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_token = self.get_scene_token(info)
        scene_frame = self.scene_frames[scene_token]
        ref_index = scene_frame.index(info)

        # NOTE: getting output frames
        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            # if this exists a valid target
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(scene_frame[curr_index], inverse=False)
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf  # pad to [4]
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T  # [3]

            # origin
            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)
        
        # select 8 origins
        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]

        return (ref_sample_token, output_origin_tensor)
