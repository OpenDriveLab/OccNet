#!/usr/bin/env bash

srun --partition=vc_research_2 \
    --mpi=pmi2 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --job-name=data \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
python tools/create_data.py nuscenes \
        --root-path ./data/nuscenes \
        --out-dir ./data/nuscenes \
        --extra-tag nuscenes \
        --version v1.0 \
        --canbus ./data
