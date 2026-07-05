# nccl
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


# used for check save when communication
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000


###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=Afford-VLA
freeze_module_list=''
base_vlm=/path/to/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/affordvla_libero.yaml
libero_data_root=/path/to/libero_affordance_plus_action
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=train_affordvla_libero_second_stage
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=offline
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Stage 2: train the full VLA with straight-through top-k region pooling.
# Set pretrained_checkpoint to the stage-1 checkpoint path.
accelerate launch \
  --config_file affordvla/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29500 \
  --num_processes 4 \
  affordvla/training/train_affordvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.affordance_head.region_pooling_mode_train hard_topk_st_pred \
  --framework.affordance_head.region_pooling_mode_infer hard_topk_pred \
  --framework.affordance_head.mask_loss_weight 0.5 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 140000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --trainer.pretrained_checkpoint /path/to/stage1_checkpoint.pt \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Afford-VLA \
  --wandb_entity your_wandb_entity
  
