# Usage of the Config

We provide the configs of BEVNet, VoxelNet and OccNet

```
BEVNet: projects/configs/bevformer 
VoxelNet: projects/configs/voxelformer
OccNet: projects/configs/hybrid
```

# Train

Train OccNet with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/hybrid/hybrid_tiny_occ.py 8
```

# Test
Eval BEVFormer with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/hybrid/hybrid_tiny_occ.py ./path/to/ckpts.pth 8
```