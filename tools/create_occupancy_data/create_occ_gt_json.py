"""
generate json file to save the occ gt info
"""
import json
import mmcv 
import os 
import tqdm
import shutil
def generate_occ_gt_json(root_dir, data_type, file_name):
    """only extract 5 scenes"""
    data_path = os.path.join(root_dir, file_name)
    
    data = mmcv.load(data_path)
    data_infos = data['infos']
    results = {}
    for index in tqdm.tqdm(range(len(data_infos))):
        info = data_infos[index]
        scene_name = info['scene_name']
        frame_idx = info['frame_idx']
        if scene_name not in results:
            results[scene_name] = {}
        target_occ_gt_dir = os.path.join(save_dir, data_type, scene_name)
        os.makedirs(target_occ_gt_dir, exist_ok=True)
        target_occ_gt_path = os.path.join(target_occ_gt_dir, '{:03d}_occ.npy'.format(frame_idx))
        target_flow_gt_path = os.path.join(target_occ_gt_dir, '{:03d}_flow.npy'.format(frame_idx))
        target_occ_invalid_path = os.path.join(target_occ_gt_dir, '{:03d}_occ_invalid.npy'.format(frame_idx))
        results[scene_name][info['token']] = {'frame_idx': frame_idx,
                                              'occ_gt_path': target_occ_gt_path,
                                              'flow_gt_path': target_flow_gt_path,
                                              'occ_invalid_path': target_occ_invalid_path}
        if os.path.exists(info['occ_gt_path']):
            shutil.copy(info['occ_gt_path'], target_occ_gt_path)
        if os.path.exists(info['flow_gt_path']):
            shutil.copy(info['flow_gt_path'], target_flow_gt_path)
        if os.path.exists(info['occ_invalid_path']):
            shutil.copy(info['occ_invalid_path'], target_occ_invalid_path)

    json_path = os.path.join(save_dir, f'occ_gt_{data_type}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)


root_dir = './data/nuscenes_occupancy_gt_normal'
save_dir = './data/occ_gt_release_v0_1'  # collect occ gt data to release
save_invalid = False
os.makedirs(save_dir, exist_ok=True)
for data_type in ['train', 'val']:
    print('process data type:', data_type)
    file_name = f'nuscenes_infos_temporal_{data_type}_occ_gt.pkl'
    generate_occ_gt_json(root_dir, data_type, file_name)