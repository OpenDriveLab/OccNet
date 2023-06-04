"""
compare occ gt and prediction
-----------
|   rgb   |
-----------
| OCC GT| OCC PRE |
| FLOW GT| FLOW PRE |
------------
"""
import numpy as np
from mayavi import mlab
import os 
import mmcv
import sys
import os
import imageio
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
mlab.options.offscreen = True

num_classes = 16
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
occ_resolution ='coarse'
if occ_resolution == 'coarse':
    occupancy_size = [0.5, 0.5, 0.5]
    voxel_size = 0.5
else:
    occupancy_size = [0.2, 0.2, 0.2]
    voxel_size = 0.2

occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
voxel_num = occ_xdim*occ_ydim*occ_zdim
add_ego_car = True

occ_colors_map = np.array(
        [   
            [255, 158, 0, 255],  #  1 car  orange
            [255, 99, 71, 255],  #  2 truck  Tomato
            [255, 140, 0, 255],  #  3 trailer  Darkorange
            [255, 69, 0, 255],  #  4 bus  Orangered
            [233, 150, 70, 255],  #  5 construction_vehicle  Darksalmon
            [220, 20, 60, 255],  #  6 bicycle  Crimson
            [255, 61, 99, 255],  #  7 motorcycle  Red
            [0, 0, 230, 255],  #  8 pedestrian  Blue
            [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
            [112, 128, 144, 255],  #  10 barrier  Slategrey
            [0, 207, 191, 255],  # 11  driveable_surface  nuTonomy green  
            [175, 0, 75, 255],  #  12 other_flat  
            [75, 0, 75, 255],  #  13  sidewalk 
            [112, 180, 60, 255],  # 14 terrain  
            [222, 184, 135, 255], # 15 manmade Burlywood 
            [0, 175, 0, 255],  # 16 vegetation  Green
            [0, 0, 0, 255],  # unknown
        ]
    ).astype(np.uint8)


def generate_the_ego_car():
    ego_range = [-2, -1, -1.5, 2, 1, 0]
    ego_voxel_size=[0.5, 0.5, 0.5]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim*ego_ydim*ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_x, ego_point_y, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*num_classes).astype(np.uint8)
    ego_points_flow = np.zeros((ego_point_xyz.shape[0], 2))
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    ego_dict['flow'] = ego_points_flow  

    return ego_dict


def obtain_points_label(occ):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    occ = np.ones(voxel_num, dtype=np.int8)*11
    occ[occ_index[:]] = occ_cls  # (voxel_num)
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim*occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])
    
    points = np.stack(points)
    points_label = occ_cls
    return points, points_label


def obtain_points_label_flow(occ, flow):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim*occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])
    
    points = np.stack(points)
    labels = occ_cls
    flow_values = flow  # 每个点具体的flow值大小

    """
    粗略区分flow的labels: 0-8
    x-> right, y->front 
    0: 静止，其余为运动的车辆
    1: front  2: left  3: back  4: right 
    """
    flow_labels = np.zeros_like(labels).astype(np.uint8)
    flow_thred = 0.5
    for i in range(len(flow_labels)):
        flow = flow_values[i]
        vel_x, vel_y = flow 
        flow_magnitude = np.linalg.norm(flow)
        if flow_magnitude < flow_thred:
            flow_labels[i] = 0
        else:
            theta = np.arctan2(vel_y, vel_x)*180/np.pi  # [-180, 180] 
            theta = int(theta + 360)%360
            if 0<= theta < 45 or 315 <= theta <=360:
                flow_labels[i] = 4
            elif 45 <= theta < 135:
                flow_labels[i] = 1
            elif 135 <= theta < 225:
                flow_labels[i] = 2
            else:
                flow_labels[i] = 3

    return points, labels, flow_values, flow_labels



def visualize_occ(points, labels, ego_dict):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    

    point_colors = np.zeros(points.shape[0])
    for cls_index in range(num_classes):
        class_point = labels == cls_index
        point_colors[class_point] = cls_index+1 

    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    lidar_plot = mlab.points3d(x, y, z, point_colors,
                                scale_factor=voxel_size,
                                mode="cube",
                                scale_mode = "vector",
                                opacity=1.0,
                                vmin=1,
                                vmax=17,
                                )
    lidar_plot.module_manager.scalar_lut_manager.lut.table = occ_colors_map

    if add_ego_car:
        ego_point_xyz = ego_dict['point']
        ego_points_label = ego_dict['label']
        ego_points_flow = ego_dict['flow']

        ego_color = np.linalg.norm(ego_point_xyz, axis=-1)
        ego_color = ego_color / ego_color.max()

        ego_plot = mlab.points3d(ego_point_xyz[:, 0], ego_point_xyz[:, 1], ego_point_xyz[:, 2], 
                                ego_color, 
                                colormap="rainbow",
                                scale_factor=voxel_size,
                                mode="cube",
                                opacity=1.0,
                                scale_mode='none',
                                )

    view_type ='back_view'
    if view_type =='back_view':
        scene = figure
        scene.scene.z_plus_view()
        scene.scene.camera.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
        scene.scene.camera.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
        scene.scene.camera.clipping_range = [0.18978054185107493, 189.78054185107493]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

    save_fig = mlab.screenshot()
    mlab.close() 
    return save_fig


def visualize_flow(points, labels, flow_values, flow_labels, ego_dict):

    back_mask = np.zeros(points.shape[0]).astype(np.bool)
    for i in range(len(labels)):
        if labels[i] in {10, 11, 12, 13, 14, 15}:
            back_mask[i] = True
    back_points = points[back_mask]
    fore_points = points[back_mask == False]
    flow_labels = flow_labels[back_mask == False]

    color_x = (back_points[:, 0] - point_cloud_range[0])/(point_cloud_range[3] -point_cloud_range[0])
    color_y = (back_points[:, 1] - point_cloud_range[1])/(point_cloud_range[4] -point_cloud_range[1])
    color_z = (back_points[:, 2] - point_cloud_range[2])/(point_cloud_range[5] -point_cloud_range[2])
    back_color = np.stack((color_x, color_y, color_z), axis=-1)
    back_color = np.linalg.norm(back_color, axis=-1)

    flow_colors = np.zeros(fore_points.shape[0])
    for cls_index in range(5):
        class_point = flow_labels == cls_index
        flow_colors[class_point] = cls_index+1 
    back_points = back_points
    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    scale_factor = voxel_size
    # background 
    back_plot = mlab.points3d(back_points[:, 0], back_points[:, 1], back_points[:, 2],
                             back_color, 
                             colormap="Greys",
                             scale_factor=scale_factor,
                             mode="cube",
                             opacity=1.0,
                             scale_mode='none',
                             )


    # visualize flows
    fore_points = fore_points 
    flow_plot = mlab.points3d(fore_points[:, 0], fore_points[:, 1], fore_points[:, 2], flow_colors,
                                scale_factor=scale_factor,
                                mode="cube",
                                scale_mode = "vector",
                                opacity=1.0,
                                vmin=1,
                                vmax=5,
                                )
    flow_colors_map = np.array(
        [   
            [0, 255, 255, 255],  #  0 stationary  
            [255, 0, 0, 255],  #  1 motion front 
            [0, 255, 0, 255],  #  2 motion left 
            [0, 0, 255, 255],  #   3 motion back
            [255, 0, 255, 255],  # 4 motion right  Magenta 
            
        ]
    ).astype(np.uint8)
    flow_plot.module_manager.scalar_lut_manager.lut.table = flow_colors_map  

    # ego voxel
    if add_ego_car:
        ego_point_xyz = ego_dict['point']
        ego_points_label = ego_dict['label']
        ego_points_flow = ego_dict['flow']

        ego_color = np.linalg.norm(ego_point_xyz, axis=-1)
        ego_color = ego_color / ego_color.max()

        ego_plot = mlab.points3d(ego_point_xyz[:, 0], ego_point_xyz[:, 1], ego_point_xyz[:, 2], 
                                ego_color, 
                                colormap="rainbow",
                                scale_factor=voxel_size,
                                mode="cube",
                                opacity=1.0,
                                scale_mode='none',
                                )

    view_type ='back_view'
    if view_type =='back_view':
        scene = figure
        scene.scene.z_plus_view()
        scene.scene.camera.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
        scene.scene.camera.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
        scene.scene.camera.clipping_range = [0.18978054185107493, 189.78054185107493]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

    save_fig = mlab.screenshot()
    mlab.close()
    return save_fig


if __name__ == '__main__':
    data_dir = './model_results/bev_tiny_det_occ_flow/epoch_9/thre_0.25'
    gt_dir = 'occ_gts' 
    pred_dir = 'occ_preds'
    ego_dict = generate_the_ego_car()
    out_flow = False

    for scene_name in os.listdir(data_dir):
        print('process scene_name:', scene_name)
        if out_flow:
            save_dir = os.path.join(data_dir, scene_name, 'visualization_flow')
        else:
            save_dir = os.path.join(data_dir, scene_name, 'visualization_occ')
        os.makedirs(save_dir, exist_ok=True)

        image_dir = os.path.join(data_dir, scene_name, 'images')
        image_names = sorted(os.listdir(image_dir))
        imgs = []
        image_num = len(image_names)

        for index in range(image_num):
            image_name = image_names[index] 
            gt_occ_file_name = image_name[:5]+'_occ.npy'
            pred_occ_file_name = image_name[:5]+'_occ.npy'

            gt_flow_file_name = image_name[:5]+'_flow.npy'
            pred_flow_file_name = image_name[:5]+'_flow.npy'

            rgb_image = imageio.imread(os.path.join(image_dir, image_name))

            occ_gt = np.load(os.path.join(data_dir, scene_name, gt_dir, gt_occ_file_name))
            points, labels = obtain_points_label(occ_gt)
            occ_gt_image = visualize_occ(points, labels, ego_dict)

            occ_pred = np.load(os.path.join(data_dir, scene_name, pred_dir, pred_occ_file_name))
            points, labels = obtain_points_label(occ_pred)
            occ_pred_image = visualize_occ(points, labels, ego_dict)

            if out_flow:
                flow_gt = np.load(os.path.join(data_dir, scene_name, gt_dir, gt_flow_file_name))
                points, labels, flow_values, flow_labels = obtain_points_label_flow(occ_gt, flow_gt)
                flow_gt_image = visualize_flow(points, labels, flow_values, flow_labels, ego_dict)

                flow_pred = np.load(os.path.join(data_dir, scene_name, pred_dir, pred_flow_file_name))
                points, labels, flow_values, flow_labels = obtain_points_label_flow(occ_pred, flow_pred)
                flow_pred_image = visualize_flow(points, labels, flow_values, flow_labels, ego_dict)

            if out_flow:
                plt.figure(figsize=(12, 15))
                plt.subplot(3, 1, 1) 
                plt.axis('off')
                plt.imshow(rgb_image)
                plt.subplot(3, 2, 3)  
                plt.axis('off')
                plt.imshow(occ_gt_image)
                plt.subplot(3, 2, 4)  
                plt.axis('off')
                plt.imshow(occ_pred_image)
                plt.subplot(3, 2, 5)  
                plt.axis('off')
                plt.imshow(flow_gt_image)
                plt.subplot(3, 2, 6)  
                plt.axis('off')
                plt.imshow(flow_pred_image)

                plt.tight_layout()
                plt.subplots_adjust(left=0.0, right=1,
                                    bottom=0.0, top=1,
                                    wspace=0.1)

            else:
                plt.figure(figsize=(12, 10))
                plt.subplot(2, 1, 1) 
                plt.axis('off')
                plt.imshow(rgb_image)
                plt.subplot(2, 2, 3)  
                plt.axis('off')
                plt.imshow(occ_gt_image)
                plt.subplot(2, 2, 4)  
                plt.axis('off')
                plt.imshow(occ_pred_image)

                plt.tight_layout()
                plt.subplots_adjust(left=0.0, right=1,
                                    bottom=0.0, top=1,
                                    wspace=0.1)
            save_path = os.path.join(save_dir, '{:03}.png'.format(index))
            plt.savefig(save_path)
            img = imageio.imread(save_path)
            # print(img.shape)
            imgs.append(img)

        video_save_path = os.path.join(save_dir, 'video.mp4')
        imageio.mimsave(video_save_path, imgs, fps=5)
    
    



        

    

