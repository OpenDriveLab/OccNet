#!/usr/bin/env bash
NTASKS=200
NTASKS_PER_NODE=10

srun --partition=vc_research_2 \
    --mpi=pmi2 \
    --ntasks=${NTASKS} \
    --ntasks-per-node=${NTASKS_PER_NODE} \
    --job-name=data \
    --kill-on-bad-exit=1 \
    --quotatype=spot \
python -u tools/create_occupancy_data/create_occ_gt_from_nuscenes_data.py