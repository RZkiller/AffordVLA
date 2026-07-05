#!/bin/bash

your_ckpt=/path/to/your/checkpoint.pt
base_port=9880
gpu_id=3
export affordvla_python=your/path/to/python

CUDA_VISIBLE_DEVICES=$gpu_id ${affordvla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16