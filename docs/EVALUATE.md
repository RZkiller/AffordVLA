# Evaluation

This guide uses LIBERO as the example benchmark for evaluating a trained Afford-VLA checkpoint. The evaluation follows a client-server workflow: one process hosts the Afford-VLA policy server, and the other process runs the LIBERO simulator and sends observations to the server.

We recommend using two separate terminals:

- **Afford-VLA environment**: runs the inference server.
- **LIBERO environment**: runs the simulator client.

## Step 1: Start the Policy Server

In the first terminal, activate the Afford-VLA environment and run:

```bash
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Before running the script, update the environment-specific values near the top of `examples/LIBERO/eval_files/run_policy_server.sh`:

```bash
export affordvla_python=your/path/to/python
your_ckpt=your/path/to/checkpoint.pt
gpu_id=0
port=5694
```

## Step 2: Start the LIBERO Simulation

In the second terminal, activate the LIBERO environment and run:

```bash
bash examples/LIBERO/eval_files/eval_libero.sh
```

Before running the script, update the paths and settings near the top of `examples/LIBERO/eval_files/eval_libero.sh`:

```bash
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/path/to/libero_env/bin/python

host="127.0.0.1"
base_port=5694
your_ckpt=/path/to/your/checkpoint.pt

task_suite_name=libero_10
num_trials_per_task=50
```

**⚠️Note:** Make sure `base_port` matches the policy server port from Step 1.

