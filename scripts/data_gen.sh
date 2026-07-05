#!/usr/bin/env bash
set -euo pipefail

# End-to-end LIBERO data generation pipeline for Afford-VLA.
#
# Pipeline:
#   1. Convert raw LeRobot LIBERO datasets to per-step offline datasets.
#   2. Generate affordance masks with scripts/batch_affordance_gen.py.
#   3. Merge mask paths back into LeRobot parquet files.
#
# Note:
#   The AffordanceVLM implementation/checkpoint is external and is not included
#   in this repository. This script only calls scripts/batch_affordance_gen.py.
#   Model-specific options are intentionally left to that Python script's
#   defaults to keep this shell entrypoint minimal.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----------------------------- User settings ----------------------------- #
RAW_LIBERO_ROOT="${RAW_LIBERO_ROOT:-/path/to/libero}"
PERSTEP_ROOT="${PERSTEP_ROOT:-/path/to/libero_per_frame}"
MASK_ROOT="${MASK_ROOT:-/path/to/ragnet_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/libero_affordance_plus_action}"

SUBSETS="${SUBSETS:-libero_object libero_spatial libero_goal libero_10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# ------------------------------------------------------------------------- #

subset_dir_name() {
  local suite="$1"
  printf "%s_no_noops_1.0.0_lerobot" "${suite}"
}

require_raw_subset() {
  local subset_dir="$1"
  if [[ ! -d "${subset_dir}" ]]; then
    echo "ERROR: raw subset directory not found: ${subset_dir}" >&2
    exit 1
  fi
}

echo "Raw LIBERO root : ${RAW_LIBERO_ROOT}"
echo "Per-step root   : ${PERSTEP_ROOT}"
echo "Mask root       : ${MASK_ROOT}"
echo "Output root     : ${OUTPUT_ROOT}"
echo "Subsets         : ${SUBSETS}"
echo

mkdir -p "${PERSTEP_ROOT}" "${MASK_ROOT}" "${OUTPUT_ROOT}"

echo "========== 1/3 Convert LeRobot datasets to per-step format =========="
for suite in ${SUBSETS}; do
  subset_name="$(subset_dir_name "${suite}")"
  src_dir="${RAW_LIBERO_ROOT}/${subset_name}"
  tgt_dir="${PERSTEP_ROOT}/${suite}_converted"

  require_raw_subset "${src_dir}"

  echo "[convert] ${suite}"
  python "${PROJECT_ROOT}/scripts/convert_libero_to_perstep.py" \
    --src_dir "${src_dir}" \
    --tgt_dir "${tgt_dir}" \
    --dataset_name "${suite}" \
    --num_workers "${NUM_WORKERS}"
done
echo

echo "========== 2/3 Generate affordance masks =========="
for suite in ${SUBSETS}; do
  data_dir="${PERSTEP_ROOT}/${suite}_converted"
  save_dir="${MASK_ROOT}/${suite}"

  if [[ ! -d "${data_dir}/episodes" ]]; then
    echo "ERROR: per-step dataset not found: ${data_dir}" >&2
    exit 1
  fi

  echo "[affordance] ${suite}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python "${PROJECT_ROOT}/scripts/batch_affordance_gen.py" \
    --data_dir "${data_dir}" \
    --save_dir "${save_dir}"
done
echo

echo "========== 3/3 Merge mask paths into parquet files =========="
subset_args=()
for suite in ${SUBSETS}; do
  subset_name="$(subset_dir_name "${suite}")"
  require_raw_subset "${RAW_LIBERO_ROOT}/${subset_name}"
  subset_args+=("${subset_name}")
done

merge_cmd=(
  python "${PROJECT_ROOT}/scripts/merge_affordance_to_parquet.py"
  --src_dir "${RAW_LIBERO_ROOT}"
  --mask_dir "${MASK_ROOT}"
  --output_dir "${OUTPUT_ROOT}"
  --num_workers "${NUM_WORKERS}"
  --subsets "${subset_args[@]}"
)

merge_cmd+=(--copy_videos)

"${merge_cmd[@]}"

echo
echo "Data generation finished."
echo "Use this path for training: ${OUTPUT_ROOT}"
