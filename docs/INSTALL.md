# Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/RZkiller/AffordVLA.git
cd AffordVLA
```

## Step 2: Set Up Python Environment

Create and activate a conda environment for Afford-VLA:

```bash
# Create a conda environment
conda create -n affordvla python=3.10 -y
conda activate affordvla

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2 with a version compatible with your PyTorch and CUDA versions
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install Afford-VLA in editable mode
pip install -e .
```

## Step 3: Download Pretrained Model Weights

Afford-VLA uses `Qwen3-VL-4B-Instruct` as the VLM backbone and uses `RAGNet` for affordance annotation.

- `Qwen3-VL-4B-Instruct`: [link🤗](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- `RAGNet`: [link🤗](https://huggingface.co/wudongming/AffordanceVLM)

After downloading the VLM backbone, set `framework.qwenvl.base_vlm` in `examples/LIBERO/train_files/affordvla_libero.yaml` to your local Qwen3-VL path.
