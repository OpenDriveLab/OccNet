# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
"""
generate train and valiation pkls from semantic kitti
"""
import os 
import os.path as osp 
import mmcv 
import glob 
import numpy as np 
import numba 

@numba.jit(nopython=True)
def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
    """Convert voxel grid coordinates to world coordinates."""
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    #    print(np.min(vox_coords))
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    
    for i in range(vox_coords.shape[0]):
        for j in range(3):
            cam_pts[i, j] = (
                vol_origin[j]
                + (vox_size * vox_coords[i, j])
                + vox_size * offsets[j]
            )
    return cam_pts

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]

@numba.jit(nopython=True)
def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in range(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix
    
def vox2pix(cam_E, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    """
    compute the 2D projection of voxels centroids
    
    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))

    return projected_pix, fov_mask, pix_z

def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out
    
    
def create_temporal_dataset(root_dir, label_dir, save_dir, splits):
    
    for split in splits:  # train or validation
        sequences = splits[split]
        dataset_infos = []
        for i in range(len(sequences)):
            scene_name = sequences[i] 
            print('process scene:', scene_name)
            
            scene_timestamp = 10000*int(scene_name)
            
            scene_token = scene_name
            scene_dir = osp.join(root_dir, scene_name)
            voxel_dir = osp.join(label_dir, scene_name)
            
            voxel_paths = sorted(glob.glob(f'{voxel_dir}/*_1_1.npy'))
            frame_nums = len(voxel_paths)
            for index in range(frame_nums):
                voxel_path = voxel_paths[index]
                frame_idx = int(voxel_path.split('/')[-1].split('_')[0])
                timestamp = scene_timestamp+frame_idx
            
                info = {
                'frame_idx': frame_idx,  # temporal related info
                'cams': dict(),
                'scene_token': scene_token,  # temporal related info
                'scene_name': scene_name,  # additional info
                'timestamp': timestamp,
                'occ_gt_path': voxel_path,
                }   
                
                camera_types = ['image_2']
                
                for cam in camera_types:
                    cam_info = {}
                    cam_info['data_path'] = osp.join(scene_dir, cam, '{:06d}.png'.format(frame_idx))
                    calib_path = osp.join(scene_dir, 'calib.txt')
                    calib_info = read_calib(calib_path)
                    cam_intrinsic = calib_info['P2'][0:3, 0:3]
                    cam_info['cam_intrinsic'] = cam_intrinsic
                    cam_info['lidar2cam'] = calib_info['Tr']
                    
                    T_velo_2_cam = calib_info['Tr']
                    cam_k = cam_intrinsic
                    
                    cam_info['T_velo_2_cam'] = T_velo_2_cam
                    cam_info['cam_k'] = cam_k
                    
                    if split == 'val':
                        vox_origin = np.array([0, -25.6, -2])
                        scene_size = (51.2, 51.2, 6.4)
                        voxel_size = 0.2 
                        img_W = 1220
                        img_H = 370
                        scale_3d = 1
                        projected_pix, fov_mask, pix_z = vox2pix(
                            T_velo_2_cam,
                            cam_k,
                            vox_origin,
                            voxel_size * scale_3d,
                            img_W,
                            img_H,
                            scene_size,
                        )            
                        # cam_info["projected_pix_{}".format(scale_3d)] = projected_pix
                        # cam_info["pix_z_{}".format(scale_3d)] = pix_z
                        cam_info["fov_mask_{}".format(scale_3d)] = fov_mask
                    info['cams'].update({cam: cam_info})
                dataset_infos.append(info)

        save_path = os.path.join(save_dir, 'kitti_infos_temporal_{}_occ_gt.pkl'.format(split))
        metadata = dict(version='v1.0-trainval')
        save_data = dict(infos=dataset_infos, metadata=metadata)
        mmcv.dump(save_data, save_path)


if __name__ == '__main__':
    root_dir = './data/kitti/dataset/sequences'
    label_dir = './data/kitti/preprocess/labels'
    save_dir = './data/kitti'
    splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            # "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    }
    create_temporal_dataset(root_dir, label_dir, save_dir, splits)