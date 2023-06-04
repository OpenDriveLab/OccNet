"""
merge 3D detection dataset and occupancy gt data
"""
import mmcv 
import os 
import json 
import tqdm
def merge(root_dir, occ_gt_dir):
    for data_type in ['train', 'val']:
        data_path = f'{root_dir}/nuscenes_infos_temporal_{data_type}.pkl'
        json_path = f'{occ_gt_dir}/occ_gt_{data_type}.json'
        occ_data = json.load(open(json_path))
        data = mmcv.load(data_path)
        data_infos = data['infos']
        save_infos = []
        for index in tqdm.tqdm(range(len(data_infos))):
            info = data_infos[index]
            scene_name = info['scene_name']
            token=info['token']
            info['occ_gt_path'] = occ_data[scene_name][token]['occ_gt_path']
            info['flow_gt_path'] = occ_data[scene_name][token]['flow_gt_path']
            info['occ_invalid_path'] = occ_data[scene_name][token]['occ_invalid_path']
            save_infos.append(info)

        save_path = os.path.join(occ_gt_dir, 'nuscenes_infos_temporal_{}_occ_gt.pkl'.format(data_type))
        metadata = dict(version='v1.0-trainval')
        save_data = dict(infos=save_infos, metadata=metadata)
        mmcv.dump(save_data, save_path)

if __name__ == '__main__':
    root_dir = 'data/nuscenes'
    occ_git_dir = 'data/occ_gt_release_v1_0'
    merge(root_dir, occ_git_dir)