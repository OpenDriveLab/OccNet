""""
save the six camera images for visualization
"""
import mmcv 
import os 
import cv2
import shutil
import tqdm

def save_camera_image(root_dir, data_type, file_name):
    """only extract 5 scenes"""
    data_path = os.path.join(root_dir, file_name)
    
    data = mmcv.load(data_path)
    data_infos = data['infos']
    
    for index in tqdm.tqdm(range(len(data_infos))):
        info = data_infos[index]
        cam_info = info['cams']
        scene_name = info['scene_name']
        frame_idx = info['frame_idx']
        for cam_type in cam_info:
            image_save_dir = os.path.join(root_dir, data_type, scene_name, 'mono_images', cam_type)
            os.makedirs(image_save_dir, exist_ok=True)
            source_image_path = cam_info[cam_type]['data_path']
            target_image_path = os.path.join(image_save_dir, '{:03d}.jpg'.format(frame_idx))
            shutil.copy(source_image_path, target_image_path)

root_dir = './data/nuscenes_occupancy_gt_normal'
for data_type in ['train', 'val']:
    print('process data type:', data_type)
    file_name = f'nuscenes_infos_temporal_{data_type}_occ_gt.pkl'
    save_camera_image(root_dir, data_type, file_name)