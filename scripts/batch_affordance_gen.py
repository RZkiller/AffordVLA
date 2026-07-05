"""
Batch affordance mask generation for per-step datasets.

Reads a per-step dataset (converted by convert_libero_to_perstep.py) and
generates affordance masks for every image_primary.jpg and image_wrist.jpg
using AffordanceVLM.

Input structure:
    {data_dir}/
    ├── meta_info.h5
    └── episodes/
        └── {episode_id:06d}/
            └── steps/
                └── {step_id:04d}/
                    ├── other.h5           # language_instruction
                    ├── image_primary.jpg
                    └── image_wrist.jpg

Output structure:
    {save_dir}/
    └── episodes/
        └── {episode_id:06d}/
            └── steps/
                └── {step_id:04d}/
                    ├── image_primary_mask.png   # binary 0/255
                    └── image_wrist_mask.png

Usage:
        CUDA_VISIBLE_DEVICES=1 python scripts/batch_affordance_gen.py \
        --data_dir /path/to/libero_spatial_converted \
        --save_dir /path/to/save_dir
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.AffordanceVLM import AffordanceVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Batch affordance mask generation for per-step datasets"
    )
    # Model arguments (same as chat.py)
    parser.add_argument("--version", default="/path/to/AffordanceNet/ckpts/AffordanceVLM-7B")
    parser.add_argument(
        "--precision", default="bf16", type=str,
        choices=["fp32", "bf16", "fp16"],
    )
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type", default="llava_v1", type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    # Batch processing arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root of per-step dataset (contains episodes/)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for masks")
    parser.add_argument("--prompt_template", type=str,
                        default="{}",
                        help="Template wrapping language_instruction. Use {} as placeholder.")
    parser.add_argument("--start_episode", type=int, default=None,
                        help="First episode index to process (inclusive)")
    parser.add_argument("--end_episode", type=int, default=None,
                        help="Last episode index to process (exclusive)")
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def load_model(args):
    """Load tokenizer and model, identical to chat.py."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens("[AFF]")
    args.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    model = AffordanceVLMForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        aff_token_idx=args.aff_token_idx,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    return model, tokenizer, clip_image_processor, transform


def build_prompt(text: str, args) -> str:
    """Build the full conversation prompt from a text query."""
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + "You are an embodied robot. " + text
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "[AFF].")
    return conv.get_prompt()


def infer_single_image(
    image_path: str,
    prompt_str: str,
    model,
    tokenizer,
    clip_image_processor,
    transform,
    args,
) -> "np.ndarray | None":
    """Run inference on a single image. Returns binary mask (H, W) uint8 0/255 or None."""
    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"  [WARNING] Cannot read image: {image_path}")
        return None
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    # CLIP preprocessing
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    # SAM preprocessing
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    # Tokenize
    input_ids = tokenizer_image_token(prompt_str, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id)


    # Prefill inference (single forward pass instead of autoregressive generation)
    h, w = original_size_list[0]
    labels = input_ids.clone()
    offset = torch.LongTensor([0, 1]).cuda()
    masks_list = [torch.zeros(1, h, w).float().cuda()]
    label_list = [torch.zeros(h, w).long().cuda()]

    with torch.no_grad():
        output_dict = model(
            images=image,
            images_clip=image_clip,
            input_ids=input_ids,
            labels=labels,
            attention_masks=attention_masks,
            offset=offset,
            masks_list=masks_list,
            label_list=label_list,
            resize_list=resize_list,
            inference=True,
        )

    pred_masks = output_dict["pred_masks"]

    # Merge all predicted masks via union (logical OR)
    merged = np.zeros((h, w), dtype=bool)
    has_mask = False
    for pred_mask in pred_masks:
        if pred_mask.shape[0] == 0:
            continue
        mask_np = pred_mask.detach().cpu().numpy()[0]  # (H, W)
        merged |= (mask_np > 0)
        has_mask = True

    if not has_mask:
        return None

    return (merged.astype(np.uint8) * 255)


def read_language_instruction(h5_path: str) -> str:
    """Read language_instruction from other.h5."""
    with h5py.File(h5_path, "r") as f:
        instr = f["language_instruction"][()]
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        return str(instr)


def main(args):
    args = parse_args(args)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    episodes_dir = data_dir / "episodes"
    if not episodes_dir.is_dir():
        print(f"Error: episodes directory not found at {episodes_dir}")
        sys.exit(1)

    # Collect and sort episode directories
    episode_dirs = sorted(
        [d for d in episodes_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    # Filter by episode range
    if args.start_episode is not None or args.end_episode is not None:
        start = args.start_episode if args.start_episode is not None else 0
        end = args.end_episode if args.end_episode is not None else len(episode_dirs)
        episode_dirs = [
            d for d in episode_dirs
            if start <= int(d.name) < end
        ]

    print(f"Data dir : {data_dir}")
    print(f"Save dir : {save_dir}")
    print(f"Episodes : {len(episode_dirs)}")
    print(f"Prompt   : {args.prompt_template}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer, clip_image_processor, transform = load_model(args)
    print("Model loaded.\n")

    total_steps = 0
    empty_mask_count = 0

    for ep_dir in episode_dirs:
        episode_id = ep_dir.name  # e.g. "000000"
        steps_dir = ep_dir / "steps"
        if not steps_dir.is_dir():
            print(f"  [WARNING] No steps/ in {ep_dir}, skipping.")
            continue

        step_dirs = sorted(
            [d for d in steps_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )

        for step_dir in step_dirs:
            step_id = step_dir.name  # e.g. "0000"

            # Read language instruction
            other_h5 = step_dir / "other.h5"
            if not other_h5.exists():
                print(f"  [WARNING] Missing other.h5 in {step_dir}, skipping.")
                continue
            language_instruction = read_language_instruction(str(other_h5))
            

            # Build prompt
            query_text = args.prompt_template.format(language_instruction)
            prompt_str = build_prompt(query_text, args)

            # Output directory (same structure as input: episodes/{episode_id}/steps/{step_id}/)
            out_dir = save_dir / "episodes" / episode_id / "steps" / step_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Process both cameras
            for cam_name in ("image_primary", "image_wrist"):
                img_path = step_dir / f"{cam_name}.jpg"
                mask_path = out_dir / f"{cam_name}_mask.png"

                if not img_path.exists():
                    print(f"  [WARNING] Missing {img_path}, skipping.")
                    continue

                mask = infer_single_image(
                    str(img_path), prompt_str,
                    model, tokenizer, clip_image_processor, transform, args,
                )

                if mask is None:
                    # Save blank mask and warn
                    h, w = cv2.imread(str(img_path)).shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    empty_mask_count += 1

                cv2.imwrite(str(mask_path), mask)

            total_steps += 1
            if total_steps % 50 == 0:
                print(f"  Processed {total_steps} steps (episode {episode_id}, step {step_id})")

        print(f"Episode {episode_id} done ({len(step_dirs)} steps)")

    print(f"\nFinished. {total_steps} steps processed, {empty_mask_count} empty masks.")


if __name__ == "__main__":
    main(sys.argv[1:])
