from typing import Optional

import torch
from affordvla.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716
)


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        # ---- AFF placeholder token (hook-based injection) ----
        '''Define the AFF placeholder symbol 🔍 and ensure it is a single token in the tokenizer.'''
        self._aff_placeholder_token = "🔍"
        _tok = self.processor.tokenizer
        _ids_bare = _tok.encode(self._aff_placeholder_token, add_special_tokens=False)
        assert len(_ids_bare) == 1, (
            f"AFF placeholder '{self._aff_placeholder_token}' must be a single token, got {_ids_bare}"
        )
        self._aff_placeholder_token_id = _ids_bare[0] # token id of the placeholder
        
        '''Verify that after concatenating placeholder 🔍 with the prompt, the tokenizer encodes the correct token id and count.'''
        _ids_ctx = _tok.encode("t🔍🔍", add_special_tokens=False)
        _ctx_hits = [t for t in _ids_ctx if t == self._aff_placeholder_token_id]
        assert len(_ctx_hits) == 2, (
            f"AFF placeholder id {self._aff_placeholder_token_id} appears "
            f"{len(_ctx_hits)}x in 't🔍🔍' → {_ids_ctx}; "
            f"tokenizer merges non-space context with emoji — choose a different placeholder"
        )

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs
    
    def get_vision_features_before_projector(self, pixel_values, image_grid_thw):
        """
        Extract image features output by the ViT blocks BEFORE the merger (projector).

        Args:
            pixel_values (torch.Tensor): Preprocessed image patches from processor,
                shape typically [total_patches, C].
            image_grid_thw (torch.LongTensor): Tiling metadata [N_images, 3]
                (temporal / height / width grid sizes).

        Returns:
            torch.Tensor: Pre-projector vision features, shape [total_patches, vision_hidden_dim].
        """

        visual = getattr(self.model, "visual", None)
        if visual is None:
            visual = self.model.model.visual

        captured = {}

        def _hook(module, args, output):
            # args[0] is the tensor entering the merger (pre-projector features)
            captured["pre_proj"] = args[0].detach()

        hook_handle = visual.merger.register_forward_hook(_hook, with_kwargs=False)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = visual(
                    pixel_values,          # positional arg: "hidden_states" in Qwen3
                    grid_thw=image_grid_thw,
                )
        finally:
            hook_handle.remove()

        return captured["pre_proj"]

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output


    # Inject actual AFF query embeddings into placeholder positions via a hook after get_input_embeddings, then forward through the LLM.
    def forward_with_aff_queries(
        self,
        aff_queries: torch.Tensor,
        aff_placeholder_positions: torch.Tensor,
        **model_inputs,
    ):
        """
        Forward VLM with AFF query embeddings injected via embedding hook.

        Args:
            aff_queries: [B, K, C] learnable AFF query embeddings.
            aff_placeholder_positions: [B, K] int64 positions of placeholder
                tokens in the input_ids sequence.
            **model_inputs: standard Qwen model inputs (input_ids,
                attention_mask, pixel_values, image_grid_thw, ...).

        Returns:
            Model outputs (with ``output_hidden_states=True``).
        """
        B, K, C = aff_queries.shape

        def inject_aff_hook(module, inputs, output):
            """Replace placeholder embeddings with learnable AFF queries."""
            out = output.clone()
            batch_idx = (
                torch.arange(B, device=out.device)
                .unsqueeze(1)
                .expand(-1, K)
            )  # [B, K]
            out[batch_idx, aff_placeholder_positions, :] = aff_queries.to(
                dtype=out.dtype, device=out.device
            )
            return out

        embedding_layer = self.model.model.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_aff_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()

        return outputs

    
    # Append AFF placeholders directly to the end of the prompt and return their positions for AFF query injection in forward_with_aff_queries.
    def build_qwenvl_inputs(self, images, instructions, solutions=None, aff_placeholder_count: int = 0, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        Args:
            aff_placeholder_count: number of AFF placeholder tokens to append
                to each prompt. When > 0, the returned dict will contain an
                extra key ``aff_placeholder_positions`` of shape [B, K].
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            # ---- Insert AFF placeholder tokens (hook-based injection) ----
            if aff_placeholder_count > 0:
                assert self._aff_placeholder_token not in prompt, (
                    f"Prompt already contains placeholder '{self._aff_placeholder_token}', cannot inject AFF tokens"
                )
                # Append placeholders directly to the end of the prompt with no leading space.
                prompt = prompt.rstrip() + self._aff_placeholder_token * aff_placeholder_count

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see affordvla/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see affordvla/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        # ---- Scan for AFF placeholder positions ----
        if aff_placeholder_count > 0:
            input_ids = batch_inputs["input_ids"]  # [B, L]
            B = input_ids.shape[0]
            aff_positions_list = []
            for b in range(B):
                pos = (input_ids[b] == self._aff_placeholder_token_id).nonzero(as_tuple=False).squeeze(-1)
                assert pos.numel() == aff_placeholder_count, (
                    f"Sample {b}: expected {aff_placeholder_count} placeholder tokens, found {pos.numel()}"
                )
                aff_positions_list.append(pos)
            aff_positions = torch.stack(aff_positions_list, dim=0)  # [B, K]
            # Verify strictly increasing
            assert torch.all(aff_positions[:, 1:] > aff_positions[:, :-1]), (
                "AFF placeholder positions must be strictly increasing"
            )
            batch_inputs["aff_placeholder_positions"] = aff_positions

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse

    # import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/affordvla_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./examples/LIBERO/train_files/affordvla_libero.yaml"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
