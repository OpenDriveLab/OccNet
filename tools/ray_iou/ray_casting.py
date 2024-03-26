import os
import time
import copy
import math
import gzip
import pickle
import argparse

import numpy as np
import cv2
import torch
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader
from tqdm import tqdm

from ego_pose_extractor import EgoPoseDataset

color_map = np.array([
    [0, 150, 245, 255],  # car                  blue
    [160, 32, 240, 255],  # truck                purple
    [135, 60, 0, 255],  # trailer              brown
    [255, 255, 0, 255],  # bus                  yellow
    [0, 255, 255, 255],  # construction_vehicle cyan
    [255, 192, 203, 255],  # bicycle              pink
    [200, 180, 0, 255],  # motorcycle           dark orange
    [255, 0, 0, 255],  # pedestrian           red
    [255, 240, 150, 255],  # traffic_cone         light yellow
    [255, 120, 50, 255],  # barrier              orangey
    [255, 0, 255, 255],  # driveable_surface    dark pink
    [175,   0,  75, 255],       # other_flat           dark red
    [75, 0, 75, 255],  # sidewalk             dard purple
    [150, 240, 80, 255],  # terrain              light green
    [230, 230, 250, 255],  # manmade              white
    [0, 175, 0, 255],  # vegetation           green
    [255, 255, 255, 255],  # free             white
], dtype=np.uint8)

occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

VIZ = False
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4


def occ2img(semantics):
    H, W, D = semantics.shape

    free_id = len(occ_class_names) - 1
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id

    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]

    viz = color_map[semantics_2d]
    viz = viz[..., :3]
    viz = cv2.resize(viz, dsize=(800, 800))

    return viz


def viz_pcd(pcd, cls):
    pcd = copy.deepcopy(pcd.astype(np.float32))
    pcd[..., 0] -= _pc_range[0]
    pcd[..., 1] -= _pc_range[1]
    pcd[..., 2] -= _pc_range[2]
    pcd[..., 0:3] /= _voxel_size
    pcd = pcd.astype(np.int32)
    pcd[..., 0] = np.clip(pcd[..., 0], a_min=0, a_max=200-1)
    pcd[..., 1] = np.clip(pcd[..., 1], a_min=0, a_max=200-1)
    pcd[..., 2] = np.clip(pcd[..., 2], a_min=0, a_max=16-1)

    free_id = len(occ_class_names) - 1
    pcd_dense = np.ones([200, 200, 16], dtype=np.int32) * free_id

    pcd_dense[pcd[..., 0], pcd[..., 1], pcd[..., 2]] = cls.astype(np.int32)

    return occ2img(pcd_dense)


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []
    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size

    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred, return_xyz=False):
    T = output_origin.shape[1]
    pred_pcds_t = []

    free_id = len(occ_class_names) - 1 
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = occ_pred.permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
    
    for t in range(T): 
        lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, 16, 200, 200],
                "test"
            )
            pred_dist *= _voxel_size

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist[0].cpu().numpy()
        )
        coord_index = coord_index[0, :, :].long().cpu()  # [N, 3]

        pred_flow = torch.from_numpy(flow_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]])  # [N, 2]
        pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
        pred_dist = pred_dist[0, :, None].cpu()

        if return_xyz:
            pred_pcds = torch.cat([pred_label, pred_dist, pred_flow, pred_pcds[0]], dim=-1)  # [N, 5]  5: [label, dist, x, y, z]
        else:
            pred_pcds = torch.cat([pred_label, pred_dist, pred_flow], dim=-1)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)

    return pred_pcds_t.numpy()


def main(args):
    token2path = {}

    data_infos = pickle.load(open(args.data_info, 'rb'))['infos']
    for info in data_infos:
        # get reletive path
        occ_path = info['occ_path'].split('nuscenes/')[-1]
        token2path[info['token']] = os.path.join(args.data_root, occ_path)

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    ego_pose_dataset = EgoPoseDataset(data_infos, dataset_type='openocc_v2')
    data_loader_kwargs={
        "pin_memory": False,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 0,
    }

    data_loader = DataLoader(
        ego_pose_dataset,
        **data_loader_kwargs,
    )

    data_pkl_gt = {}
    data_pkl_pred = {}

    for batch in tqdm(data_loader, ncols=50):
        sample_token = batch[0][0]
        output_origin = batch[1].to(torch.float32)

        pred_filepath = os.path.join(args.pred_root, sample_token + '.npz')
        pred_data = np.load(pred_filepath, allow_pickle=True)
        '''sem_pred = pred_data['semantics']
        sem_pred = np.reshape(sem_pred, [200, 200, 16])
        sem_pred = torch.from_numpy(sem_pred)
        flow_pred = pred_data['flow']
        flow_pred = np.reshape(flow_pred, [200, 200, 16, 2])'''
        sem_pred = pred_data['pred']
        sem_pred = np.reshape(sem_pred, [200, 200, 16])
        sem_pred = torch.from_numpy(sem_pred)
        flow_pred = np.zeros([200, 200, 16, 2], dtype=np.float32)

        gt_filepath = token2path[sample_token]
        gt_data = np.load(gt_filepath, allow_pickle=True)
        sem_gt = gt_data['semantics']
        sem_gt = torch.from_numpy(sem_gt)

        flow_gt = gt_data['flow']
        flow_gt = np.reshape(flow_gt, [200, 200, 16, 2])

        pcd_gt = process_one_sample(sem_gt, lidar_rays, output_origin, flow_gt, return_xyz=VIZ)
        pcd_pred = process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred, return_xyz=VIZ)

        if VIZ:
            pcdimg = viz_pcd(pcd_pred[:, 4:], pcd_pred[:, 0])
            os.makedirs('vis', exist_ok=True)
            cv2.imwrite('vis/%s_pcd.jpg' % sample_token, pcdimg[..., ::-1])

        data_pkl_gt[sample_token] = {
            'pcd_cls': pcd_gt[:, 0].astype(np.uint8),
            'pcd_dist': pcd_gt[:, 1].astype(np.float16),
            'pcd_flow': pcd_gt[:, 2:4].astype(np.float16)
        }

        data_pkl_pred[sample_token] = {
            'pcd_cls': pcd_pred[:, 0].astype(np.uint8),
            'pcd_dist': pcd_pred[:, 1].astype(np.float16),
            'pcd_flow': pcd_pred[:, 2:4].astype(np.float16)
        }

    submission_pkl_gt = {
        'method': 'GT',
        'team': 'OpenDriveLab',
        'authors': 'OpenDriveLab',
        'e-mail': 'contact@opendrivelab.com',
        'institution / company': "OpenDriveLab",
        'country / region': "China",
        'results': data_pkl_gt
    }

    submission_pkl_pred = {
        'method': 'My prediction',
        'team': 'My team',
        'authors': 'Me',
        'e-mail': 'email',
        'institution / company': "Me",
        'country / region': "Earth",
        'results': data_pkl_pred
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path_gt = os.path.join(args.output_dir, os.path.basename(args.data_info).split('.')[0] + '_pcd.gz')
    output_path_pred = os.path.join(args.output_dir, 'my_pred_pcd.gz')
    print("gzip and dumping the data...")

    start = time.time()
    with gzip.GzipFile(output_path_gt, 'wb', compresslevel=9) as f:
        pickle.dump(submission_pkl_gt, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.GzipFile(output_path_pred, 'wb', compresslevel=9) as f:
        pickle.dump(submission_pkl_pred, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default='../../data/nuscenes')
    parser.add_argument("--data-info", default='../../data/nuscenes/nuscenes_infos_val_occ.pkl')
    parser.add_argument("--pred-root", default='./your_prediction')
    parser.add_argument("--output-dir", default='./output/')
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
