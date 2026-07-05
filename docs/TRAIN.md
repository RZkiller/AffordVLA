# Training

Afford-VLA follows a two-stage training paradigm. This guide uses LIBERO as the example benchmark and assumes that you have already prepared the affordance-augmented LeRobot datasets following [DATASET.md](DATASET.md).

## Step 1: Set Up the Training Config

The default LIBERO training config is provided at:

```text
examples/LIBERO/train_files/affordvla_libero.yaml
```

Before launching training, update the following paths for your local environment:

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/Qwen3-VL-4B-Instruct

datasets:
  vla_data:
    data_root_dir: /path/to/libero_affordance_plus_action
    mask_root_dir: /path/to/ragnet_results
```

The provided launch scripts also define runtime paths and logging settings. Please check the editable block near the top of each script:

```text
examples/LIBERO/train_files/run_libero_stage1.sh
examples/LIBERO/train_files/run_libero_stage2.sh
```

At minimum, verify `base_vlm`, `libero_data_root`, `run_root_dir`, `run_id`, `CUDA_VISIBLE_DEVICES`, and the Weights & Biases settings. If you change the number of GPUs, keep `--num_processes` consistent with the visible devices.

## Step 2: Stage-1 Training

Stage 1 trains the affordance-related modules while freezing the VLM, action model, and region feature projection. This gives the affordance head stable spatial grounding before its predictions are used to condition action generation.

Run:

```bash
bash examples/LIBERO/train_files/run_libero_stage1.sh
```

By default, the stage-1 script uses ground-truth affordance masks for training:

```text
--framework.affordance_head.region_pooling_mode_train hard_gt
--framework.affordance_head.region_pooling_mode_infer hard_pred
--trainer.freeze_modules qwen_vl_interface,action_model,region_feat_proj
```

Checkpoints are saved under:

```text
${run_root_dir}/${run_id_stage1}/checkpoints
```

## Step 3: Stage-2 Training

Stage 2 starts from the stage-1 checkpoint and jointly trains the affordance head, action head, and VLM. In this stage, the action head learns to use predicted affordance regions for action generation, while the action loss can also backpropagate into the affordance head. This enables action-aligned affordance learning.

Before running stage 2, set the stage-1 checkpoint path in `examples/LIBERO/train_files/run_libero_stage2.sh`:

```text
--trainer.pretrained_checkpoint /path/to/stage1_checkpoint.pt
```

Then run:

```bash
bash examples/LIBERO/train_files/run_libero_stage2.sh
```

By default, the stage-2 script enables straight-through top-k region pooling:

```text
--framework.affordance_head.region_pooling_mode_train hard_topk_st_pred
--framework.affordance_head.region_pooling_mode_infer hard_topk_pred
```

The final model is saved to:

```text
${run_root_dir}/${run_id_stage2}/final_model
```
