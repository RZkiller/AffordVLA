# Data Generation Pipeline

This directory contains the data generation utilities used to build the
Afford-VLA training data from raw LeRobot-format LIBERO datasets.

The pipeline has three stages:

1. Convert raw LeRobot episodes into a frame-level offline dataset.
2. Generate affordance masks for each frame using an external affordance
   segmentation model.
3. Merge the generated mask paths back into the original LeRobot parquet files.

The final output is a LeRobot-compatible dataset with action/state data and
additional affordance mask path columns.

## External Affordance Mask Model

Affordance mask generation uses the model and codebase from:

**AffordanceNet / RAGNet**  
https://github.com/wudongming97/AffordanceNet

RAGNet is introduced in *RAGNet: Large-scale Reasoning-based Affordance
Segmentation Benchmark towards General Grasping*.

This repository does not include the AffordanceNet/RAGNet implementation,
checkpoint, or environment setup. Please follow the official AffordanceNet
repository for installation, checkpoints, and runtime dependencies. In this
repository, `batch_affordance_gen.py` is only the caller script used to run
affordance mask inference on the converted per-frame LIBERO data.

## Expected Input

The raw LIBERO data should be in LeRobot format:

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

The default suite list is:

```text
libero_object libero_spatial libero_goal libero_10
```

## Quick Start

The recommended entrypoint is:

```bash
bash scripts/data_gen.sh
```

You can override the required paths and runtime settings with environment
variables:

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
| `RAW_LIBERO_ROOT` | Root directory of raw LeRobot LIBERO datasets | `/path/to/libero` |
| `PERSTEP_ROOT` | Output root for intermediate per-frame datasets | `/path/to/libero_per_frame` |
| `MASK_ROOT` | Output root for generated affordance masks | `/path/to/ragnet_results` |
| `OUTPUT_ROOT` | Final LeRobot dataset root with affordance mask paths | `/path/to/libero_affordance_plus_action` |
| `SUBSETS` | Space-separated LIBERO suites to process | `libero_object libero_spatial libero_goal libero_10` |
| `NUM_WORKERS` | Number of worker processes for conversion and merging | `8` |
| `CUDA_VISIBLE_DEVICES` | GPU device used by affordance mask generation | `1` |

`data_gen.sh` intentionally passes only the required dataset arguments to
`batch_affordance_gen.py`. Model-specific options for AffordanceNet/RAGNet are
kept inside `batch_affordance_gen.py` and should be configured according to the
external AffordanceNet setup.

## Pipeline Details

### 1. Convert LeRobot Data to Per-Step Data

Script:

```bash
python scripts/convert_libero_to_perstep.py
```

Example for one suite:

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

The conversion preserves frame alignment: parquet row `N` maps to per-step
directory `steps/{N:04d}`.

### 2. Generate Affordance Masks

Script:

```bash
python scripts/batch_affordance_gen.py
```

Example for one suite:

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

The generated masks are binary PNG files. Each mask path is later stored in the
merged parquet files as a relative path under `MASK_ROOT`.

### 3. Merge Mask Paths into Parquet Files

Script:

```bash
python scripts/merge_affordance_to_parquet.py
```

Example:

```bash
python scripts/merge_affordance_to_parquet.py \
  --src_dir /path/to/libero \
  --mask_dir /path/to/ragnet_results \
  --output_dir /path/to/libero_affordance_plus_action \
  --num_workers 8 \
  --copy_videos
```

This stage adds two string columns to each parquet file:

```text
affordance_mask.primary
affordance_mask.wrist
```

Example values:

```text
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

## Training Configuration

After generation, use:

```yaml
datasets:
  vla_data:
    data_root_dir: /path/to/libero_affordance_plus_action
    mask_root_dir: /path/to/ragnet_results
```

`data_root_dir` points to the merged LeRobot dataset. `mask_root_dir` points to
the root directory containing generated mask files. The parquet files store mask
paths relative to `mask_root_dir`.

## Notes

- Do not use partial episode ranges when merging masks into parquet unless you
  fully understand the alignment assumptions. The merge stage assumes each
  parquet row has a corresponding generated mask at the same step index.
- Check the merge script output and make sure the reported number of missing
  masks is `0`.
- If your AffordanceNet/RAGNet code lives outside this repository, ensure its
  Python modules are visible before running `batch_affordance_gen.py`, for
  example by activating the correct environment or setting `PYTHONPATH`.
