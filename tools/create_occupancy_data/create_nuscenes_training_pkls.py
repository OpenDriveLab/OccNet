"""
merge 700 train or 150 val scenes to generate pkls
"""
import mmcv
import os

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
        save_path = os.path.join(root_dir, 'nuscenes_infos_temporal_{}_occ_gt.pkl'.format(data_type))
        metadata = dict(version='v1.0-trainval')
        save_data = dict(infos=datas, metadata=metadata)
        mmcv.dump(save_data, save_path)

root_dir = './data/nuscenes_occupancy_gt_normal'
merge_scene(root_dir)
