# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu

import math
import copy
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tqdm import tqdm
from prettytable import PrettyTable

dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])

_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4
_occ_size = [200, 200, 16]

occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

flow_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian',
]

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


def process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred):
    # lidar origin in ego coordinate
    # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
    T = output_origin.shape[1]
    pred_pcds_t = []

    free_id = len(occ_class_names) - 1 
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = torch.from_numpy(occ_pred).permute(2, 1, 0)
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
        coord_index = coord_index[0, :, :].int().cpu()  # [N, 3]

        pred_flow = torch.from_numpy(flow_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]])  # [N, 2]
        pred_label = torch.from_numpy(sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]])[:, None]  # [N, 1]
        pred_dist = pred_dist[0, :, None].cpu()
        pred_pcds = torch.cat([pred_label.float(), pred_dist, pred_flow], dim=-1)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)
   
    return pred_pcds_t.numpy()


def calc_metrics(pcd_pred_list, pcd_gt_list):
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

    ave = np.zeros([len(thresholds), len(occ_class_names)])
    for i, cls in enumerate(occ_class_names):
        if cls not in flow_class_names:
            ave[:, i] = np.nan

    ave_count = np.zeros([len(thresholds), len(occ_class_names)])

    for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
        for j, threshold in enumerate(thresholds):
            # L1
            depth_pred = pcd_pred[:, 1]
            depth_gt = pcd_gt[:, 1]
            l1_error = np.abs(depth_pred - depth_gt)
            tp_dist_mask = (l1_error < threshold)
            
            for i, cls in enumerate(occ_class_names):
                cls_id = occ_class_names.index(cls)
                cls_mask_pred = (pcd_pred[:, 0] == cls_id)
                cls_mask_gt = (pcd_gt[:, 0] == cls_id)

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()

                # flow L2 error
                if cls in flow_class_names and tp_mask.sum() > 0:
                    gt_flow_i = pcd_gt[tp_mask, 2:4]
                    pred_flow_i = pcd_pred[tp_mask, 2:4]
                    flow_error = np.linalg.norm(gt_flow_i - pred_flow_i, axis=1)
                    ave[j][i] += np.sum(flow_error)
                    ave_count[j][i] += flow_error.shape[0]
    
    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    ave_list = ave[1][:-1] / ave_count[1][:-1]  # use threshold = 2
    
    return iou_list, ave_list


def main(sem_pred_list, sem_gt_list, flow_pred_list, flow_gt_list, lidar_origin_list):
    torch.cuda.empty_cache()

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    pcd_pred_list, pcd_gt_list = [], []
    for sem_pred, sem_gt, flow_pred, flow_gt, lidar_origins in tqdm(zip(sem_pred_list, sem_gt_list, flow_pred_list, flow_gt_list, lidar_origin_list), ncols=50):
        sem_pred = np.reshape(sem_pred, [200, 200, 16])
        sem_gt = np.reshape(sem_gt, [200, 200, 16])
        flow_pred = np.reshape(flow_pred, [200, 200, 16, 2])
        flow_gt = np.reshape(flow_gt, [200, 200, 16, 2])

        pcd_pred = process_one_sample(sem_pred, lidar_rays, lidar_origins, flow_pred)
        pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins, flow_gt)

        # evalute on non-free rays
        valid_mask = (pcd_gt[:, 0].astype(np.int32) != len(occ_class_names) - 1)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_list.append(pcd_pred)
        pcd_gt_list.append(pcd_gt)

    iou_list, ave_list = calc_metrics(pcd_pred_list, pcd_gt_list)
    
    table = PrettyTable([
        'Class Names',
        'IoU@1', 'IoU@2', 'IoU@4', 'AVE'
    ])
    table.float_format = '.3'

    for i in range(len(occ_class_names) - 1):
        table.add_row([
            occ_class_names[i],
            iou_list[0][i], iou_list[1][i], iou_list[2][i], ave_list[i]
        ], divider=(i == len(occ_class_names) - 2))
    
    table.add_row([
        'MEAN',
        np.nanmean(iou_list[0]),
        np.nanmean(iou_list[1]),
        np.nanmean(iou_list[2]),
        np.nanmean(ave_list)
    ])

    print(table)

    miou = np.nanmean(iou_list)
    mave = np.nanmean(ave_list)
    
    occ_score = miou * 0.9 + max(1 - mave, 0.0) * 0.1
    
    print(' --- Occ score:', occ_score)

    torch.cuda.empty_cache()
