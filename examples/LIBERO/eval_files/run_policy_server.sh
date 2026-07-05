#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export affordvla_python=your/path/to/python
your_ckpt=your/path/to/checkpoint.pt
gpu_id=0
port=5694
################# start Policy Server ######################

CUDA_VISIBLE_DEVICES=$gpu_id ${affordvla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
