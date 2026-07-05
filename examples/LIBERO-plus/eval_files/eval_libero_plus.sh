#!/bin/bash
# unset PYTHONPATH
# unset LD_LIBRARY_PATH

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/path/to/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/path/to/libero_plus_env/bin/python
export MUJOCO_GL=osmesa
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

base_port=9896
host="127.0.0.1"
unnorm_key="franka"
gpu_id=0
your_ckpt=/path/to/your/checkpoint.pt

# Task suite to evaluate. Options: libero_goal libero_spatial libero_object libero_10
# Set to "all" to evaluate all 4 suites at once
TASK_SUITES="libero_10"

# Pre-generate folder name from checkpoint path for output directory naming
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# Generate a unified timestamp to keep log times consistent across tasks
timestamp=$(date +"%Y%m%d_%H%M%S")
# === End of environment variable configuration ===
###########################################################################################

# Set up a trap to handle SIGINT and SIGTERM signals for cleanup
trap 'echo -e "\n[!] Interrupt received, killing background processes..."; kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; echo "Cleanup complete, exiting."; exit 1' SIGINT SIGTERM

# export DEBUG=true

if [ "$TASK_SUITES" = "all" ]; then
    TASK_SUITES="libero_goal libero_spatial libero_object libero_10"
fi

for task_suite_name in ${TASK_SUITES}; do
    output_dir="results_plus/${task_suite_name}/${folder_name}"
    LOG_DIR="${output_dir}/logs/${timestamp}"
    mkdir -p "${LOG_DIR}"

    video_out_path="${output_dir}/videos"
    log_file="${LOG_DIR}/${task_suite_name}.log"

    echo "============================================"
    echo "  >>> Task suite     : ${task_suite_name}"
    echo "  >>> Inference port : ${base_port}"
    echo "  >>> Output dir     : ${output_dir}"
    echo "  >>> Start time     : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=$gpu_id ${LIBERO_Python} ./examples/LIBERO-plus/eval_files/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task 1 \
        --args.video-out-path "$video_out_path" \
        --args.log-path "$LOG_DIR" \
        2>&1 | tee "${log_file}" &
done

# =============== wait for all evaluation tasks to finish ===============
echo "Waiting for all evaluation tasks to finish..."
wait