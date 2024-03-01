# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu

import torch
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val, test


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/data/nusc.py#L22
class nuScenesDataset(Dataset):
    def __init__(self, nusc, nusc_split):
        """
        Figure out a list of sample data tokens for training.
        """
        super(nuScenesDataset, self).__init__()

        self.nusc = nusc
        self.nusc_split = nusc_split
        self.nusc_root = self.nusc.dataroot

        scenes = self.nusc.scene

        if self.nusc_split == "train":
            split_scenes = train
        elif self.nusc_split == "val":
            split_scenes = val
        else:
            split_scenes = test

        # list all sample data
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_tokens = []
        self.sample_data_tokens = []
        self.timestamps = []

        for scene in scenes:
            if scene["name"] not in split_scenes:
                continue
            scene_token = scene["token"]
            # location
            log = self.nusc.get("log", scene["log_token"])
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if log["location"].startswith("singapore") else False
            #
            start_index = len(self.sample_tokens)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_token = first_sample["token"]
            i = 0
            while sample_token != "":
                self.flip_flags.append(flip_flag)
                self.scene_tokens.append(scene_token)
                self.sample_tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                i += 1
                self.timestamps.append(sample["timestamp"])
                sample_data_token = sample["data"]["LIDAR_TOP"]

                self.sample_data_tokens.append(sample_data_token)
                sample_token = sample["next"]
            
            end_index = len(self.sample_tokens)
             
            valid_start_index = start_index 
            valid_end_index = end_index 
            self.valid_index += list(range(valid_start_index, valid_end_index))

        assert len(self.sample_tokens) == len(self.scene_tokens) == len(self.flip_flags) == len(self.timestamps)

    def __len__(self):
        return len(self.valid_index)

    def get_global_pose(self, sd_token, inverse=False):
        sd = self.nusc.get("sample_data", sd_token)
        sd_ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        sd_cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        if inverse is False:
            global_from_ego = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
            )
            ego_from_sensor = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
            )
            pose = global_from_ego.dot(ego_from_sensor)
        else:
            sensor_from_ego = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
            )
            ego_from_global = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
            )
            pose = sensor_from_ego.dot(ego_from_global)

        return pose

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]

        ref_sample_token = self.sample_tokens[ref_index]
        ref_scene_token = self.scene_tokens[ref_index]
        ref_sd_token = self.sample_data_tokens[ref_index]  # sample["data"]["LIDAR_TOP"]
        flip_flag = self.flip_flags[ref_index]

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)

        # NOTE: getting output frames
        output_origin_list = []

        for curr_index in range(len(self.valid_index)):
            # if this exists a valid target
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            elif self.scene_tokens[curr_index] == ref_scene_token:
                curr_sd_token = self.sample_data_tokens[curr_index]

                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)

                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)
            else:
                continue

            #  lidar2ego
            ref_sd = self.nusc.get("sample_data", ref_sd_token)
            cs_record = self.nusc.get('calibrated_sensor', ref_sd['calibrated_sensor_token'])
            l2e_r = cs_record['rotation']
            l2e_t = cs_record['translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix

            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = l2e_r_mat
            lidar2ego[:3, -1] = l2e_t
            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf  # pad to [4]
            origin_tf = np.dot(lidar2ego[:3], origin_tf_pad.T).T  # [3]

            # origin
            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)
        
        # select 8 origins
        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]
      
        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]

        return (ref_sample_token, output_origin_tensor)