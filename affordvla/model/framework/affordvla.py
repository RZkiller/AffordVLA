import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from affordvla.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from affordvla.model.framework.base_framework import baseframework
from affordvla.model.modules.vlm import get_vlm_model
from affordvla.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from affordvla.training.trainer_utils.trainer_tools import resize_images
from affordvla.model.tools import FRAMEWORK_REGISTRY
from affordvla.model.modules.affordance_head import RegionPooling
from affordvla.model.modules.affordance_head import AffordanceHead



@FRAMEWORK_REGISTRY.register("Afford-VLA")
class Afford_VLA(baseframework):

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()

        # Instantiate QwenVL interface and Action Model
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size


        # ---- Hard Region Pooling ----
        self.region_pooling = RegionPooling(num_sample_point=256)
        # Project region-pooled vision features into the same space as Qwen hidden dim
        vision_hidden_dim = self.qwen_vl_interface.model.config.vision_config.hidden_size
        llm_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.region_feat_proj = nn.Linear(vision_hidden_dim, llm_hidden_dim)


        # ---- Initialize AFF Query Tokens ----
        aff_cfg = config.framework.get("affordance_head", {})
        self.num_aff_queries_per_view = aff_cfg.get("num_queries_per_view", 4)
        # Define per-view AFF query tokens here
        self.aff_queries = nn.Parameter(
            torch.zeros(1, self.num_aff_queries_per_view, llm_hidden_dim)
        )
        nn.init.normal_(self.aff_queries, std=0.02)


        # ---- Initialize Affordance Head ----
        self.affordance_head = AffordanceHead(
            llm_hidden_dim=llm_hidden_dim,
            vision_hidden_dim=vision_hidden_dim,
            decoder_dim=aff_cfg.get("decoder_dim", 256),
            num_decoder_layers=aff_cfg.get("num_decoder_layers", 2),
            num_decoder_heads=aff_cfg.get("num_decoder_heads", 8),
            num_aff_queries_per_view=self.num_aff_queries_per_view,
            base_grid_size=tuple(aff_cfg.get("base_grid_size", [16, 16])), 
        )
        self.mask_loss_weight = aff_cfg.get("mask_loss_weight", 1.0)

        # ---- View-Aware AFF: per-view identity embedding ----
        self.use_view_aware_aff = aff_cfg.get("use_view_aware_aff", True)
        self.max_num_views = aff_cfg.get("max_num_views", 2)
        # zero init ensures behavior matches baseline after loading old checkpoint with strict=False
        self.aff_view_embed = nn.Parameter(
            torch.zeros(1, self.max_num_views, llm_hidden_dim)
        )

        # ---- Hard Top-K Patch Pooling + Straight-Through ----
        self.region_pooling_mode_train = aff_cfg.get("region_pooling_mode_train", "hard_topk_st_pred")
        self.region_pooling_mode_infer = aff_cfg.get("region_pooling_mode_infer", "hard_topk_pred")
        self.topk_k = aff_cfg.get("topk_k", 16)
        self.topk_ratio = aff_cfg.get("topk_ratio", None)
        self.topk_tau_bwd = aff_cfg.get("topk_tau_bwd", 1.0)

        # Freeze aff_view_embed updates in stage 2
        if self.region_pooling_mode_train == "hard_topk_st_pred" and self.use_view_aware_aff:
            self.aff_view_embed.requires_grad_(False)

    
    # ------------------------------------------------------------------ #
    #                  Region Pooling Helper                              #
    # ------------------------------------------------------------------ #
    def _compute_region_feat(
        self,
        pre_proj_feat: torch.Tensor,
        image_grid_thw: torch.Tensor,
        affordance_masks: List,
        num_images_per_batch: List[int],
    ) -> Optional[torch.Tensor]:
        """
        Compute region features via region pooling on pre-projector vision
        features with affordance masks, then project to LLM hidden dim.
        """
        device = pre_proj_feat.device
        B = len(num_images_per_batch)

        # Split pre_proj_feat per image & record spatial shapes
        per_image_feats: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []
        offset = 0
        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            num_patches = int(t * h * w)
            per_image_feats.append(pre_proj_feat[offset : offset + num_patches])
            spatial_shapes.append((int(h * t), int(w)))  # collapse t into h
            offset += num_patches

        # Per-batch-item region pooling
        img_offset = 0
        all_region_feats: List[torch.Tensor] = []
        for b in range(B):
            n_imgs = num_images_per_batch[b]
            batch_feats  = per_image_feats[img_offset : img_offset + n_imgs]
            batch_masks  = affordance_masks[img_offset : img_offset + n_imgs]
            batch_shapes = spatial_shapes[img_offset : img_offset + n_imgs]
            img_offset += n_imgs

            region_feat_list = self.region_pooling(
                batch_feats,
                batch_masks,
                original_dtype=pre_proj_feat.dtype,
                return_dtype=pre_proj_feat.dtype,
                spatial_shapes=batch_shapes,
            )
            # Each view yields exactly [1, 1, C_vision] after pooling;
            # cat across views -> [n_views, 1, C_vision], squeeze -> [n_views, C_vision]
            cat_feat = torch.cat(region_feat_list, dim=0).squeeze(1)  # [n_views, C_vision]
            all_region_feats.append(cat_feat)

        # Stack across batch -> [B, n_views, C_vision], then project
        stacked = torch.stack(all_region_feats, dim=0)        # [B, n_views, C_vision]
        stacked = stacked.to(dtype=self.region_feat_proj.weight.dtype)
        region_feat_proj = self.region_feat_proj(stacked)      # [B, n_views, C_llm]
        return region_feat_proj

    # Extract affordance_mask for each image in the batch into a flat list
    def _build_flat_masks(self, examples: List[dict]) -> List[List[torch.Tensor]]:
        """
        Build flat affordance_masks list aligned with image order.

        Assumes every sample carries ``affordance_mask`` and each view has
        exactly one mask tensor.

        Returns:
            flat list of length N_total_images, each element is a length-1
            list ``[mask_tensor]``.
        """
        flat_masks: List[List[torch.Tensor]] = []
        for example in examples:
            masks_per_view = example["affordance_mask"]
            n_views = len(example["image"])
            assert len(masks_per_view) == n_views, (
                f"affordance_mask length ({len(masks_per_view)}) must match "
                f"image count ({n_views}) per sample"
            )
            for m in masks_per_view:
                if isinstance(m, torch.Tensor):
                    flat_masks.append([m])  # wrap single mask into list
                else:
                    raise TypeError(
                        f"Expected affordance_mask element to be torch.Tensor, got {type(m)}"
                    )
        return flat_masks
        

    # ------------------------------------------------------------------ #
    #    AffordanceHead Helpers                                          #
    # ------------------------------------------------------------------ #
    def _get_patch_grid_hw(self, image_grid_thw: torch.Tensor) -> Tuple[int, int]:
        """Extract patch grid (H_p, W_p) from image_grid_thw[0]."""
        t, h, w = image_grid_thw[0].tolist()
        return (int(h * t), int(w))

    def _reshape_pre_proj_to_vision_patches(
        self,
        pre_proj_feat: torch.Tensor,
        image_grid_thw: torch.Tensor,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """
        Reshape flat pre_proj_feat [total_patches=B*V*Np, C_vis] into [B, V, Np, C_vis].

        Requires all images to have the same patch count.
        """
        N_total = image_grid_thw.shape[0] # N_total=B*num_views, total number of images
        assert N_total == B * V, (
            f"Expected {B}*{V}={B*V} images, got {N_total}"
        )
        patches_per = [int(thw[0] * thw[1] * thw[2]) for thw in image_grid_thw]
        assert all(p == patches_per[0] for p in patches_per), (
            f"All views must have same patch count, got {patches_per}"
        )
        Np = patches_per[0] # number of patches per image
        return pre_proj_feat.reshape(B, V, Np, -1)

    def _build_flat_masks_from_predicted(
        self,
        mask_logits: torch.Tensor,
        patch_grid_hw: Tuple[int, int],
        threshold: float = 0.5,
    ) -> List[List[torch.Tensor]]:
        """
        Convert predicted mask_logits [B, V, H_p, W_p] into flat mask list
        compatible with _compute_region_feat.

        Returns:
            flat list of length B*V, each element is [mask_tensor] with
            shape [H_p, W_p].
        """
        B, V, H_p, W_p = mask_logits.shape
        binary = (mask_logits.sigmoid() > threshold).to(mask_logits.dtype)
        flat_masks: List[List[torch.Tensor]] = []
        for b in range(B):
            for v in range(V):
                flat_masks.append([binary[b, v]])  # [H_p, W_p]
        return flat_masks

    def _compute_mask_loss(
        self,
        mask_logits: torch.Tensor,
        examples: List[dict],
        patch_grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute BCE loss between predicted mask_logits and GT masks.
        """
        B, V, H_p, W_p = mask_logits.shape
        assert patch_grid_hw == (H_p, W_p), (
            f"Patch grid from image_grid_thw {patch_grid_hw} must match "
            f"mask_logits spatial dims {(H_p, W_p)}"
        )
        gt_tensor = self._downsample_gt_masks(examples, patch_grid_hw)  # [B, V, H_p, W_p]
        gt_tensor = gt_tensor.to(device=mask_logits.device)
        # Compute in float32 for numerical stability
        return F.binary_cross_entropy_with_logits(
            mask_logits.float(), gt_tensor.float()
        )

    def _downsample_gt_masks(
        self,
        examples: List[dict],
        patch_grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Downsample GT affordance masks to patch grid resolution.

        Returns:
            gt_ds: [B, V, H_p, W_p] float tensor (0~1 range from area interpolation).
        """
        H_p, W_p = patch_grid_hw
        gt_list: List[torch.Tensor] = []
        for example in examples:
            masks_per_view = example["affordance_mask"]
            n_views = len(example["image"])
            assert len(masks_per_view) == n_views, (
                f"affordance_mask length ({len(masks_per_view)}) must match "
                f"image count ({n_views}) per sample"
            )
            for m in masks_per_view:
                if not isinstance(m, torch.Tensor):
                    raise TypeError(
                        f"Expected affordance_mask element to be torch.Tensor, got {type(m)}"
                    )
                m_float = m.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H_orig, W_orig]
                m_ds = F.interpolate(m_float, size=(H_p, W_p), mode="area")
                gt_list.append(m_ds.squeeze())  # [H_p, W_p]
        B = len(examples)
        V = len(examples[0]["affordance_mask"])
        return torch.stack(gt_list).reshape(B, V, H_p, W_p)

    # ------------------------------------------------------------------ #
    #    Hard Top-K Patch Pooling + Straight-Through Gradient              #
    # ------------------------------------------------------------------ #
    def _resolve_topk_k(self, Np: int) -> int:
        """
        Resolve the actual k value for top-k patch selection.

        Priority: self.topk_k > self.topk_ratio * Np.
        Result is clamped to [1, Np].
        """
        if self.topk_k is not None:
            k = int(self.topk_k)
            if k <= 0:
                raise ValueError(f"topk_k must be positive, got {self.topk_k}")
        elif self.topk_ratio is not None:
            r = float(self.topk_ratio)
            if not (0.0 < r <= 1.0):
                raise ValueError(f"topk_ratio must be in (0, 1], got {self.topk_ratio}")
            k = int(round(r * Np))
        else:
            raise ValueError("Either topk_k or topk_ratio must be set for hard top-k pooling.")
        if self.topk_tau_bwd <= 0:
            raise ValueError(f"topk_tau_bwd must be positive, got {self.topk_tau_bwd}")
        return max(1, min(k, Np))

    def _build_hard_topk_mask(
        self,
        mask_logits: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build hard top-k binary patch mask from mask_logits.

        Args:
            mask_logits: [B, V, H_p, W_p] raw logits from AffordanceHead.
            k: number of top patches to select per view.

        Returns:
            m_hard: [B, V, Np] binary mask (1 for selected, 0 otherwise).
            topk_idx: [B, V, k] indices of selected patches.
        """
        B, V, H_p, W_p = mask_logits.shape
        z = mask_logits.reshape(B, V, -1)  # [B, V, Np]
        topk_idx = torch.topk(z, k=k, dim=-1, largest=True, sorted=False).indices  # [B, V, k]
        m_hard = torch.zeros_like(z)
        m_hard.scatter_(-1, topk_idx, 1.0)  # [B, V, Np]
        return m_hard, topk_idx

    def _build_st_topk_mask(
        self,
        mask_logits: torch.Tensor,
        k: int,
        tau_bwd: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build straight-through top-k mask for training.

        Forward value equals m_hard (hard top-k selection).
        Backward gradient flows through softmax surrogate q.

        Args:
            mask_logits: [B, V, H_p, W_p]
            k: number of top patches per view.
            tau_bwd: temperature for backward surrogate softmax.

        Returns:
            m_st: [B, V, Np] — forward == m_hard, backward via q.
            m_hard: [B, V, Np] — pure hard mask (detached).
        """
        B, V, H_p, W_p = mask_logits.shape
        z = mask_logits.reshape(B, V, -1)  # [B, V, Np]

        # Hard top-k
        m_hard, _ = self._build_hard_topk_mask(mask_logits, k)

        # Cast to float32 explicitly to avoid softmax overflow under bf16
        q = torch.softmax(z.float() / tau_bwd, dim=-1) * float(k)  # [B, V, Np] fp32
        q = q.to(z.dtype)  # Restore original dtype to match subsequent einsum

        # Straight-through: forward = m_hard, backward = q
        m_st = m_hard + q - q.detach()
        return m_st, m_hard

    def _compute_hard_topk_region_feat(
        self,
        vision_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Compute region features by averaging top-k selected patches.

        Args:
            vision_patches: [B, V, Np, C_vis] patch-aligned vision features.
            patch_mask: [B, V, Np] selection mask (hard or ST).
            k: number of selected patches (used as divisor).

        Returns:
            region_feat: [B, V, C_llm] projected region features.
        """
        patch_mask = patch_mask.to(dtype=vision_patches.dtype)
        pooled = torch.einsum('bvn,bvnc->bvc', patch_mask, vision_patches) / float(k)  # [B, V, C_vis]
        region_feat = self.region_feat_proj(
            pooled.to(dtype=self.region_feat_proj.weight.dtype)
        )  # [B, V, C_llm]
        return region_feat

    # ------------------------------------------------------------------ #
    #    View-Aware AFF Query Construction                                #
    # ------------------------------------------------------------------ #
    def _build_view_aware_aff_queries(
        self,
        batch_size: int,
        n_views: int,
    ) -> torch.Tensor:
        """
        Build AFF queries with explicit per-view identity.

        Formula: Q[v, k] = Q_base[k] + E_view[v]

        Returns:
            aff_queries: [B, V*K, C_llm]
                flatten order (view-first):
                [view0_k0, ..., view0_kK-1, view1_k0, ..., view1_kK-1, ...]
        """
        if n_views > self.max_num_views:
            raise ValueError(
                f"n_views={n_views} exceeds max_num_views={self.max_num_views}. "
                f"Please increase affordance_head.max_num_views in config."
            )

        # [1, 1, K, C] + [1, V, 1, C] -> broadcast -> [1, V, K, C]
        base_q = self.aff_queries[:, None, :, :]
        view_e = self.aff_view_embed[:, :n_views, None, :]
        aff_q = base_q + view_e

        # [1, V*K, C] -> [B, V*K, C]
        K = self.num_aff_queries_per_view
        aff_q = aff_q.reshape(1, n_views * K, -1).expand(batch_size, -1, -1)
        return aff_q


    # ------------------------------------------------------------------ #
    #    AFF Query Token Definition and VLM Injection                    #
    # ------------------------------------------------------------------ #
    def _forward_vlm_with_aff(
        self,
        qwen_inputs: dict,
        n_views: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward VLM with AFF query tokens injected via embedding hook,
        then split outputs into VLM hidden states and AFF hidden states.

        Args:
            qwen_inputs: dict from build_qwenvl_inputs (must include
                ``aff_placeholder_positions`` key of shape [B, K_total]).
            n_views: number of views per sample (used to expand aff_queries).

        Returns:
            vlm_hidden: [B, L - K_total, C_llm] VLM hidden states with
                AFF positions removed.
            aff_hidden: [B, K_total, C_llm] hidden states at AFF positions.
        """
        # Shallow copy to avoid mutating the caller's dict
        model_inputs = dict(qwen_inputs)
        aff_positions = model_inputs.pop("aff_placeholder_positions")  # [B, K_total = n_views * K]

        B = model_inputs["input_ids"].shape[0]
        K_total = self.num_aff_queries_per_view * n_views

        # Stability checks
        assert aff_positions.shape == (B, K_total), (
            f"aff_positions shape {aff_positions.shape} != expected ({B}, {K_total})"
        )
        assert torch.all(aff_positions[:, 1:] > aff_positions[:, :-1]), (
            "aff_placeholder_positions must be strictly increasing"
        )

        # Build AFF queries: view-aware (shared base + per-view embed) or baseline (repeat)
        if self.use_view_aware_aff:
            aff_queries = self._build_view_aware_aff_queries(B, n_views)
        else:
            aff_queries = self.aff_queries.repeat(1, n_views, 1).expand(B, -1, -1)

        outputs = self.qwen_vl_interface.forward_with_aff_queries(
            aff_queries=aff_queries,
            aff_placeholder_positions=aff_positions,
            **model_inputs,
        )

        last_hidden = outputs.hidden_states[-1]  # [B, L, C]
        L = last_hidden.shape[1]

        # Extract AFF hidden states by position
        batch_idx = torch.arange(B, device=last_hidden.device).unsqueeze(1).expand(-1, K_total)
        aff_hidden = last_hidden[batch_idx, aff_positions]  # [B, K_total, C]

        # Build boolean mask to remove AFF positions and get vlm_hidden
        aff_mask = torch.zeros(B, L, dtype=torch.bool, device=last_hidden.device)
        aff_mask[batch_idx, aff_positions] = True
        # All samples have the same non-AFF length since placeholder count is fixed
        vlm_len = L - K_total
        vlm_hidden = last_hidden[~aff_mask].reshape(B, vlm_len, -1)  # [B, L - K_total, C]

        return vlm_hidden, aff_hidden


    # ------------------------------------------------------------------ #
    #                       Forward (training)                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """

        # =======================Extract Data============================ #
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # =====================Build QWenVL input format===================== #
        n_views = len(batch_images[0])
        K_total = self.num_aff_queries_per_view * n_views
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, aff_placeholder_count=K_total,
        )


        # =====================Extract Qwen vision-encoder features ===================== #
        pixel_values  = qwen_inputs["pixel_values"]        # [B*num_views*hw, C]
        image_grid_thw = qwen_inputs["image_grid_thw"]     # [N_images=B*num_views, 3]
        pre_proj_feat = self.qwen_vl_interface.get_vision_features_before_projector(  # [B*num_views*patches_perimage, C_vision]
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        

        # ====================VLM feedforward to get last-layer hidden states==================== #
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_hidden, aff_hidden = self._forward_vlm_with_aff(qwen_inputs, n_views=n_views)


        # ====================Affordance Head & Region Pooling=================== #
        B = len(examples)
        mask_loss = None
        mask_logits = None
        last_hidden = vlm_hidden

        # Check affordance mask availability
        all_have_mask = all(
            example.get("affordance_mask") is not None for example in examples
        )
        none_have_mask = all(
            example.get("affordance_mask") is None for example in examples
        )
        if not all_have_mask and not none_have_mask:
            raise ValueError(
                "Inconsistent affordance_mask presence: all samples must have it or none should have it."
            )

        # Run affordance head forward and region pooling
        if all_have_mask:
            patch_grid_hw = self._get_patch_grid_hw(image_grid_thw) # Get patch grid size
            
            # Affordance Head forward to get mask_logits
            vision_patches = self._reshape_pre_proj_to_vision_patches(
                pre_proj_feat, image_grid_thw, B, n_views,
            ) # [B, num_views, num_patches, C_Vison]
            mask_logits = self.affordance_head(
                aff_hidden, n_views, patch_grid_hw, vision_patches,
            )  # [B, V, H_p, W_p]
            mask_loss = self._compute_mask_loss(mask_logits, examples, patch_grid_hw) # compute mask loss

            # Region Pooling: select path based on region_pooling_mode_train
            mode = self.region_pooling_mode_train
            if mode == "hard_gt":
                # Original teacher forcing: GT mask + hard pooling
                num_images_per_batch = [len(imgs) for imgs in batch_images]
                flat_masks = self._build_flat_masks(examples)
                region_feat = self._compute_region_feat(
                    pre_proj_feat, image_grid_thw, flat_masks, num_images_per_batch,
                )  # [B, num_views, C_llm]
            elif mode == "hard_topk_st_pred":
                # Hard top-k patch selection + straight-through gradient
                Np = vision_patches.shape[2]
                k = self._resolve_topk_k(Np)
                m_st, m_hard = self._build_st_topk_mask(
                    mask_logits, k=k, tau_bwd=self.topk_tau_bwd,
                )
                region_feat = self._compute_hard_topk_region_feat(
                    vision_patches, patch_mask=m_st, k=k,
                )
            else:
                raise ValueError(f"Unknown region_pooling_mode_train: {mode}")

            # Concatenate affordance embedding with VLM hidden states
            if region_feat is not None:
                last_hidden = torch.cat([vlm_hidden, region_feat], dim=1)  # [B, L + V, C_llm]


        # ====================Action Expert Forward and Loss=================== #
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        # total_loss
        total_loss = action_loss
        if mask_loss is not None:
            total_loss = action_loss + self.mask_loss_weight * mask_loss

        # Return total_loss (for backward) and detached individual losses (for logging)
        return {
            "total_loss": total_loss,
            "action_loss": action_loss.detach(),
            "mask_loss": mask_loss.detach() if mask_loss is not None else None,
            "mask_logits": mask_logits.detach() if mask_logits is not None else None,
        }



    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Predict actions for a batch of examples.
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        # =======================Extract Data====================== #
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # ====================Build QWenVL input format===================== #
        n_views = len(batch_images[0])
        K_total = self.num_aff_queries_per_view * n_views
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, aff_placeholder_count=K_total,
        )

        # ====================Extract Qwen vision-encoder features ===================== #
        pixel_values  = qwen_inputs["pixel_values"]        # [total_patches, C]
        image_grid_thw = qwen_inputs["image_grid_thw"]     # [N_images, 3]
        pre_proj_feat = self.qwen_vl_interface.get_vision_features_before_projector(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
    
        # ====================VLM feedforward to get last-layer hidden states==================== #
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_hidden, aff_hidden = self._forward_vlm_with_aff(qwen_inputs, n_views=n_views)


        # ====================Affordance Head: predict mask → region pooling================== #
        B = len(examples)
        patch_grid_hw = self._get_patch_grid_hw(image_grid_thw)

        vision_patches = self._reshape_pre_proj_to_vision_patches(
            pre_proj_feat, image_grid_thw, B, n_views,
        )

        mask_logits = self.affordance_head(
            aff_hidden, n_views, patch_grid_hw, vision_patches,
        )  # [B, V, H_p, W_p]

        last_hidden = vlm_hidden
        patch_topk_vis = None
        mode = self.region_pooling_mode_infer
        if mode == "hard_pred":
            num_images_per_batch = [len(imgs) for imgs in batch_images]
            flat_masks = self._build_flat_masks_from_predicted(mask_logits, patch_grid_hw)
            region_feat = self._compute_region_feat(
                pre_proj_feat, image_grid_thw, flat_masks, num_images_per_batch,
            )
        elif mode == "hard_topk_pred":
            # Hard top-k patch selection
            Np = vision_patches.shape[2]
            k = self._resolve_topk_k(Np)
            m_hard, _ = self._build_hard_topk_mask(mask_logits, k=k)
            region_feat = self._compute_hard_topk_region_feat(
                vision_patches, patch_mask=m_hard, k=k,
            )
            H_p, W_p = patch_grid_hw
            patch_topk_vis = m_hard.reshape(B, n_views, H_p, W_p).float().detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown region_pooling_mode_infer: {mode}")
        if region_feat is not None:
            last_hidden = torch.cat([vlm_hidden, region_feat], dim=1)  # [B, L + V, C_llm]

        # ========================Action Expert Forward========================== #
        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        mask_probs = torch.sigmoid(mask_logits).float().detach().cpu().numpy()  # [B, V, H_p, W_p], range [0, 1]
        result = {"normalized_actions": normalized_actions, "mask_probs": mask_probs}
        if patch_topk_vis is not None:
            result["patch_topk"] = patch_topk_vis  # [B, V, H_p, W_p]
        return result



if __name__ == "__main__":
    from omegaconf import OmegaConf
    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/affordvla_libero.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    args.config_yaml = "./examples/LIBERO/train_files/affordvla_libero.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    
    # Initialize model
    model: Afford_VLA = Afford_VLA(cfg)
    # print(model)


    # ==================== Use Fake Sample Test Forward & Predict ====================
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], 
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "affordance_mask": [torch.randint(0, 2, (224, 224), dtype=torch.uint8), torch.randint(0, 2, (224, 224), dtype=torch.uint8)] # binary mask for the image
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], 
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "affordance_mask": [torch.randint(0, 2, (224, 224), dtype=torch.uint8), torch.randint(0, 2, (224, 224), dtype=torch.uint8)] # binary mask for the image
    }
    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # forward test
    forward_output = model(batch)
    total_loss = forward_output['total_loss']
    action_loss = forward_output['action_loss']
    mask_loss = forward_output['mask_loss']
    # loss check
    print(f"Total Loss: {total_loss.item()}")
    print(f"Action Loss: {action_loss.item()}")
    print(f"Mask Loss: {mask_loss.item()}")


    # predict action test
    predict_output = model.predict_action(examples=[sample]) # state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    
    # ======================== Test Dataloader ========================
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from affordvla.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
        # shuffle=True,
    )

    
    # ================ Forward model with dataloader ================
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        break

    
