"""
train: scene num 750, sample num: 18310
val: scene num 150, sample num: 6019

extract partial data
"""
import mmcv 
import os 

def extract_mini_data(root_dir, file_name):
    """only extract 5 scenes"""
    data_path = os.path.join(root_dir, file_name)
    
    data = mmcv.load(data_path)
    data_infos = data['infos']
    metadata = data['metadata']
    scene_number = 5
    start_frame = 10
    end_frame = 30
    parital_data_infos = []

    scene_set=set()
    for i in range(len(data_infos)):
        frame_idx = data_infos[i]['frame_idx']
        scene_token = data_infos[i]['scene_token']
        scene_set.add(scene_token)
        if len(scene_set) > scene_number:
            break
        # if frame_idx >= start_frame and frame_idx < end_frame:
        parital_data_infos.append(data_infos[i])

    print('extract samples:', len(parital_data_infos))

    save_partial_data = dict(infos=parital_data_infos, metadata=metadata)

    save_path = os.path.join(root_dir, 'mini_'+file_name)
    mmcv.dump(save_partial_data, save_path)

root_dir = './data/nuscenes_occupancy_gt_normal'

for data_type in ['train', 'val']:
    print('process data type:', data_type)
    file_name = f'nuscenes_infos_temporal_{data_type}_occ_gt.pkl'
    extract_mini_data(root_dir, file_name)

