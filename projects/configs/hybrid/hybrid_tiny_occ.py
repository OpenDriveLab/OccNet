# tiny model ResNet50
# occupancy_size = 0.5

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.2, 0.2, 8]
occupancy_size = [0.5, 0.5, 0.5]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_occupancy_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200
bev_z_ = 16
queue_length = 3 # each sequence contains `queue_length` frames.
use_occ_gts = True
only_occ = True
only_det = False

# info of voxel block encoder
voxel_encoder1_dim = _dim_//2
voxel_encoder2_dim = _dim_//2
voxel_encoder3_dim = _dim_//4
voxel_encoder4_dim = _dim_//4
bev_z1 = 2
bev_z2 = 4
bev_z3 = 8
bev_z4 = 16
_pos_dim_1 = voxel_encoder1_dim//2
_pos_dim_2 = voxel_encoder2_dim//2
_pos_dim_3 = voxel_encoder3_dim//2
_pos_dim_4 = voxel_encoder4_dim//2

last_voxel_dims = voxel_encoder4_dim

decoder_on_bev = True
box_query_dims = _dim_

# decoder_on_bev = False
# box_query_dims = voxel_encoder4_dim

model = dict(
    type='HybridFormer',
    use_grid_mask=True,
    video_test_mode=True,
    use_occ_gts=use_occ_gts,
    only_occ=only_occ,
    only_det=only_det,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='HybridFormerOccupancyHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        point_cloud_range=point_cloud_range,
        occupancy_size=occupancy_size,
        occ_dims=_occupancy_dim_,
        occupancy_classes=16,
        only_occ=only_occ,
        only_det=only_det,
        last_voxel_dims=last_voxel_dims,
        box_query_dims=box_query_dims,  # detection head
        transformer=dict(
            type='HybridPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            decoder_on_bev=decoder_on_bev,
            encoder_embed_dims=[_dim_, voxel_encoder1_dim, voxel_encoder2_dim, voxel_encoder3_dim, voxel_encoder4_dim],  # the dim of cascaded voxel encoder
            feature_map_z=[1, bev_z1, bev_z2, bev_z3, bev_z4],  # the height of cascaded voxel encoder
            pos_dims=[_pos_dim_, _pos_dim_1, _pos_dim_2, _pos_dim_3, _pos_dim_4],
            position=dict(
                bev=dict(
                    type='LearnedPositionalEncoding',
                    num_feats=_pos_dim_,
                    row_num_embed=bev_h_,
                    col_num_embed=bev_w_,
                    ),
                voxel=dict(
                    type='VoxelLearnedPositionalEncoding',
                    num_feats=_pos_dim_1,
                    row_num_embed=bev_h_,
                    col_num_embed=bev_w_,
                    z_num_embed=bev_z1,
                    )
                ), 
            encoder=dict(
                bev=dict(  # the bev encoder
                    type='BEVFormerEncoder',
                    num_layers=1,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BEVFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='TemporalSelfAttention',
                                embed_dims=_dim_,
                                num_points=4,  
                                num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=_dim_,
                                    num_points=8,
                                    num_levels=_num_levels_),
                                embed_dims=_dim_,
                            )
                        ],
                        ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm'))),
                voxel=dict(  # the config of first cascaded voxel encoder
                    type='VoxelFormerEncoder',
                    num_layers=1,
                    pc_range=point_cloud_range,
                    num_points_in_voxel=4,   
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='VoxelFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='VoxelTemporalSelfAttention',
                                embed_dims=voxel_encoder1_dim,
                                num_points=4,  
                                num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=voxel_encoder1_dim,
                                    num_points=8,
                                    num_levels=_num_levels_),
                                embed_dims=voxel_encoder1_dim,
                            )
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=voxel_encoder1_dim,
                            feedforward_channels=voxel_encoder1_dim*2,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),
                            ),  
                        feedforward_channels=voxel_encoder1_dim*2,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm'))),
               ),
            ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_occupancy=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadOccupancyGT'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'occ_gts'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        use_occ_gts=use_occ_gts,   # 10 foreground + 6 background 
        data_root=data_root,
        ann_file='data/nuscenes_occupancy_gt_normal/nuscenes_infos_temporal_train_occ_gt.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file='data/nuscenes_occupancy_gt_normal/nuscenes_infos_temporal_val_occ_gt.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file='data/nuscenes_occupancy_gt_normal/nuscenes_infos_temporal_val_occ_gt.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
find_unused_parameters = False