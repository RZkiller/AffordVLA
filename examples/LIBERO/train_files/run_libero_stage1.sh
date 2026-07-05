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
# 冻结 VLM、Action Model、region_feat_proj，仅训练 affordance_head + aff_queries
freeze_module_list='qwen_vl_interface,action_model,region_feat_proj'
base_vlm=/path/to/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/affordvla_libero.yaml
libero_data_root=/path/to/libero_affordance_plus_action
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=train_affordvla_libero_first_stage
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Stage 1: train the affordance head with GT masks, then evaluate with predicted masks.

accelerate launch \
  --config_file affordvla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  affordvla/training/train_affordvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.affordance_head.region_pooling_mode_train hard_gt \
  --framework.affordance_head.region_pooling_mode_infer hard_pred \
  --framework.affordance_head.mask_loss_weight 1.0 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 4000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Afford-VLA \
  --wandb_entity your_wandb_entity
