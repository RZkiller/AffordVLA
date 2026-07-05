# Data Generation Pipeline

Afford-VLA augments the original LeRobot-format datasets with offline affordance mask annotations. This document uses the LIBERO benchmark as an example and describes how to generate the training data used by Afford-VLA.

## Prepare Raw LIBERO Dataset

Download the raw [LIBERO datasets](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset) from the Hugging Face collection. The downloaded raw LIBERO data should follow the LeRobot directory format:

```text
RAW_LIBERO_ROOT/
├── libero_object_no_noops_1.0.0_lerobot/
│   ├── data/
│   ├── meta/
│   └── videos/
├── libero_spatial_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
└── libero_10_no_noops_1.0.0_lerobot/
```

The default LIBERO suites processed by the pipeline are:

```text
libero_object libero_spatial libero_goal libero_10
```

## Pipeline Overview

The data generation pipeline has three stages.

### 1. Convert Raw LeRobot Episodes to Per-Step Data

This stage decodes the raw LeRobot parquet/video data into a frame-level offline dataset. Each original timestep is converted into a per-step directory that contains images and metadata.

Example for one LIBERO suite:

```bash
python scripts/convert_libero_to_perstep.py \
  --src_dir /path/to/libero/libero_object_no_noops_1.0.0_lerobot \
  --tgt_dir /path/to/libero_per_frame/libero_object_converted \
  --dataset_name libero_object \
  --num_workers 8
```

Output structure:

```text
PERSTEP_ROOT/libero_object_converted/
├── meta_info.h5
└── episodes/
    └── 000000/
        ├── meta_info.h5
        └── steps/
            └── 0000/
                ├── other.h5
                ├── image_primary.jpg
                └── image_wrist.jpg
```

### 2. Generate Affordance Masks

This stage generates affordance masks for each frame using an external affordance segmentation model.

Example for one LIBERO suite:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/batch_affordance_gen.py \
  --data_dir /path/to/libero_per_frame/libero_goal_converted \
  --save_dir /path/to/ragnet_results/libero_goal
```

Output structure:

```text
MASK_ROOT/libero_goal/
└── episodes/
    └── 000000/
        └── steps/
            └── 0000/
                ├── image_primary_mask.png
                └── image_wrist_mask.png
```

The generated masks are binary PNG files. Each mask path is later written into the merged parquet files as a relative path under `MASK_ROOT`.

**⚠️Note:** RAGNet is an external dependency and is not included in this repository. Please follow the official [RAGNet Repo](https://github.com/wudongming97/AffordanceNet) for environment setup and model preparation.

### 3. Merge Mask Path Back into LeRobot Parquet Files

This stage merges the generated mask paths back into the original LeRobot parquet files. The final output is still LeRobot-compatible and keeps the original action/state data, while adding affordance mask path columns.

```bash
python scripts/merge_affordance_to_parquet.py \
  --src_dir /path/to/libero \
  --mask_dir /path/to/ragnet_results \
  --output_dir /path/to/libero_affordance_plus_action \
  --num_workers 8 \
  --copy_videos
  
# This stage adds two string columns to each parquet file:
	affordance_mask.primary
	affordance_mask.wrist

# Example values:
  libero_goal/episodes/000001/steps/0042/image_primary_mask.png
  libero_goal/episodes/000001/steps/0042/image_wrist_mask.png
```

The output directory mirrors the original LeRobot dataset structure:

```text
OUTPUT_ROOT/
├── libero_object_no_noops_1.0.0_lerobot/
│   ├── data/
│   ├── meta/
│   └── videos/
├── libero_spatial_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
└── libero_10_no_noops_1.0.0_lerobot/
```

## Quick Start

Instead of running the three stages manually, you can use the provided end-to-end entrypoint:

```bash
bash scripts/data_gen.sh
```

You can override the required paths and runtime settings with environment variables:

```bash
RAW_LIBERO_ROOT=/path/to/libero \
PERSTEP_ROOT=/path/to/libero_per_frame \
MASK_ROOT=/path/to/ragnet_results \
OUTPUT_ROOT=/path/to/libero_affordance_plus_action \
CUDA_VISIBLE_DEVICES=1 \
NUM_WORKERS=8 \
bash scripts/data_gen.sh
```

Available variables:

| Variable | Description | Default |
| --- | --- | --- |
| `RAW_LIBERO_ROOT` | Root directory of the raw LeRobot-format LIBERO datasets | `/path/to/libero` |
| `PERSTEP_ROOT` | Output root for intermediate per-step datasets | `/path/to/libero_per_frame` |
| `MASK_ROOT` | Output root for generated affordance masks | `/path/to/ragnet_results` |
| `OUTPUT_ROOT` | Final LeRobot dataset root with affordance mask paths | `/path/to/libero_affordance_plus_action` |
| `SUBSETS` | Space-separated LIBERO suites to process | `libero_object libero_spatial libero_goal libero_10` |
| `NUM_WORKERS` | Number of worker processes for conversion and merging | `8` |
| `CUDA_VISIBLE_DEVICES` | GPU device used for affordance mask generation | `1` |

## Training Configuration

After generation, update the VLA dataset paths in `examples/LIBERO/train_files/affordvla_libero.yaml`:

```yaml
datasets:
  vla_data:
    data_root_dir: /path/to/libero_affordance_plus_action
    mask_root_dir: /path/to/ragnet_results
```

For the provided training scripts, also update the `libero_data_root` variable:

```bash
# examples/LIBERO/train_files/run_libero_stage1.sh
# examples/LIBERO/train_files/run_libero_stage2.sh
libero_data_root=/path/to/libero_affordance_plus_action
```

`data_root_dir` points to the merged LeRobot dataset. `mask_root_dir` points to the root directory that contains the generated mask files. The parquet files store mask paths relative to `mask_root_dir`.
