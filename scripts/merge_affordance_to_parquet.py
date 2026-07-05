"""
Merge pre-generated affordance mask paths into LeRobot parquet files.

src_dir, mask_dir, output_dir are the **root** directories that contain
multiple LIBERO subset subdirectories, e.g.:

  src_dir/
  ├── libero_10_no_noops_1.0.0_lerobot/     (data/, meta/, videos/)
  ├── libero_goal_no_noops_1.0.0_lerobot/
  ├── libero_object_no_noops_1.0.0_lerobot/
  └── libero_spatial_no_noops_1.0.0_lerobot/

  mask_dir/
  ├── libero_10/                            (episodes/...)
  ├── libero_goal/
  ├── libero_object/
  └── libero_spatial/

For each episode parquet, two new string columns are added:
  - affordance_mask.primary : path relative to mask_dir root, e.g.
        libero_goal/episodes/000001/steps/0042/image_primary_mask.png
  - affordance_mask.wrist   : same convention for wrist camera mask

The output directory mirrors the subset structure:
  output_dir/
  ├── libero_10_no_noops_1.0.0_lerobot/
  │   ├── data/          # parquet files with new columns
  │   ├── meta/          # copied from src_dir
  │   └── videos/        # symlinked from src_dir
  └── ...

Usage:
    python scripts/merge_affordance_to_parquet.py \
        --src_dir  /path/to/LEROBOT_LIBERO_DATA \
        --mask_dir /path/to/ragnet_results \
        --output_dir /path/to/LEROBOT_LIBERO_DATA_WITH_MASK \
        --num_workers 8 \
        --copy_videos

"""

import argparse
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# LeRobot metadata helpers (same as convert script)
# ---------------------------------------------------------------------------

def load_info(subset_dir: Path) -> dict:
    with open(subset_dir / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes(subset_dir: Path) -> list[dict]:
    with open(subset_dir / "meta" / "episodes.jsonl") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Auto-discover subset directories
# ---------------------------------------------------------------------------

def discover_subsets(src_dir: Path, mask_dir: Path) -> list[str]:
    """Find subset directory names that exist in both src_dir and mask_dir.

    A valid subset has meta/info.json in src_dir and episodes/ in mask_dir.
    """
    subsets = []
    for d in sorted(src_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "meta" / "info.json").exists():
            continue
        short_name = short_subset_name(d.name)
        mask_sub = mask_dir / d.name
        short_mask_sub = mask_dir / short_name
        if not mask_sub.is_dir() and not short_mask_sub.is_dir():
            print(f"  [SKIP] {d.name}: no matching mask directory found")
            continue
        subsets.append(d.name)
    return subsets


def short_subset_name(subset_name: str) -> str:
    suffix = "_no_noops_1.0.0_lerobot"
    if subset_name.endswith(suffix):
        return subset_name[: -len(suffix)]
    return subset_name


def resolve_mask_subset_name(subset_name: str, mask_dir: Path) -> str:
    """Return the mask subdirectory name used under mask_dir.

    For compatibility, both full LeRobot subset names and short LIBERO suite
    names are supported. Short names match scripts/data_gen.sh, e.g.
    ``libero_goal``.
    """
    if (mask_dir / subset_name).is_dir():
        return subset_name

    short_name = short_subset_name(subset_name)
    if (mask_dir / short_name).is_dir():
        return short_name

    return subset_name


# ---------------------------------------------------------------------------
# Per-episode merge worker
# ---------------------------------------------------------------------------

def merge_episode(args: tuple) -> tuple[str, str, int, int]:
    """Merge mask paths into a single episode parquet.

    Returns (subset_name, episode_id_str, num_steps, num_missing_masks).
    """
    (
        subset_src_dir,
        mask_root_dir,
        subset_output_dir,
        subset_name,
        mask_subset_name,
        episode_index,
        data_path_pattern,
        chunk_size,
        verify_masks,
    ) = args

    subset_src_dir = Path(subset_src_dir)
    mask_root_dir = Path(mask_root_dir)
    subset_output_dir = Path(subset_output_dir)

    episode_id_str = str(episode_index).zfill(6)
    chunk_index = episode_index // chunk_size

    # --- Read original parquet (preserves row order) ---
    parquet_rel = data_path_pattern.format(
        episode_chunk=chunk_index, episode_index=episode_index
    )
    src_parquet = subset_src_dir / parquet_rel
    assert src_parquet.exists(), f"Source parquet not found: {src_parquet}"

    df = pd.read_parquet(src_parquet)
    num_steps = len(df)

    # --- Build mask path columns ---
    # Paths are relative to mask_root_dir, including the mask subset prefix.
    primary_paths = []
    wrist_paths = []
    missing = 0

    for step_idx in range(num_steps):
        step_id_str = str(step_idx).zfill(4)
        rel_primary = f"{mask_subset_name}/episodes/{episode_id_str}/steps/{step_id_str}/image_primary_mask.png"
        rel_wrist = f"{mask_subset_name}/episodes/{episode_id_str}/steps/{step_id_str}/image_wrist_mask.png"

        if verify_masks:
            for rel_path in (rel_primary, rel_wrist):
                if not (mask_root_dir / rel_path).exists():
                    missing += 1

        primary_paths.append(rel_primary)
        wrist_paths.append(rel_wrist)

    df["affordance_mask.primary"] = primary_paths
    df["affordance_mask.wrist"] = wrist_paths

    # --- Write to output directory (same relative path) ---
    out_parquet = subset_output_dir / parquet_rel
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    return subset_name, episode_id_str, num_steps, missing


# ---------------------------------------------------------------------------
# Process a single subset
# ---------------------------------------------------------------------------

def process_subset(
    subset_name: str,
    src_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    num_workers: int,
    start_ep: int | None,
    end_ep: int | None,
    verify_masks: bool,
    copy_videos: bool,
) -> tuple[int, int, int]:
    """Process one subset (e.g. libero_goal_no_noops_1.0.0_lerobot).

    Returns (num_episodes_processed, total_steps, total_missing).
    """
    subset_src = src_dir / subset_name
    subset_out = output_dir / subset_name
    subset_out.mkdir(parents=True, exist_ok=True)

    info = load_info(subset_src)
    episodes = load_episodes(subset_src)
    data_path_pattern = info["data_path"]
    chunk_size = info["chunks_size"]
    total_episodes = len(episodes)
    mask_subset_name = resolve_mask_subset_name(subset_name, mask_dir)

    ep_start = start_ep if start_ep is not None else 0
    ep_end = end_ep if end_ep is not None else total_episodes

    print(f"  Episodes  : {ep_start} → {ep_end} (total {total_episodes})")
    print(f"  Mask dir  : {mask_subset_name}")

    # --- 1. Copy meta/ directory ---
    src_meta = subset_src / "meta"
    dst_meta = subset_out / "meta"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    shutil.copytree(src_meta, dst_meta)

    # --- 2. Symlink (or copy) videos/ directory ---
    src_videos = subset_src / "videos"
    dst_videos = subset_out / "videos"
    if src_videos.exists():
        if dst_videos.exists() or dst_videos.is_symlink():
            if dst_videos.is_symlink():
                dst_videos.unlink()
            else:
                shutil.rmtree(dst_videos)
        if copy_videos:
            shutil.copytree(src_videos, dst_videos)
        else:
            dst_videos.symlink_to(src_videos)

    # --- 3. Merge parquet files ---
    worker_args = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        if ep_idx < ep_start or ep_idx >= ep_end:
            continue
        worker_args.append((
            str(subset_src),
            str(mask_dir),        # root mask dir, not subset
            str(subset_out),
            subset_name,
            mask_subset_name,
            ep_idx,
            data_path_pattern,
            chunk_size,
            verify_masks,
        ))

    results = []
    if num_workers <= 1:
        for wa in worker_args:
            result = merge_episode(wa)
            results.append(result)
    else:
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(merge_episode, worker_args):
                results.append(result)

    n_eps = len(results)
    n_steps = sum(r[2] for r in results)
    n_missing = sum(r[3] for r in results)

    print(f"  Done: {n_eps} episodes, {n_steps} steps, {n_missing} missing masks")
    return n_eps, n_steps, n_missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge affordance mask paths into LeRobot parquet files"
    )
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Root dir containing LIBERO subset dirs (e.g. LEROBOT_LIBERO_DATA/)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Root dir containing mask result subset dirs (e.g. ragnet_results/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root dir for merged datasets")
    parser.add_argument("--subsets", type=str, nargs="*", default=None,
                        help="Subset dir names to process (default: auto-discover all)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--start_episode", type=int, default=None,
                        help="First episode index to process (inclusive)")
    parser.add_argument("--end_episode", type=int, default=None,
                        help="Last episode index to process (exclusive)")
    parser.add_argument("--no_verify_masks", action="store_true", default=False,
                        help="Skip checking whether each mask file exists")
    parser.add_argument("--copy_videos", action="store_true", default=False,
                        help="Copy videos instead of symlinking (slower but portable)")
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_masks = not args.no_verify_masks

    # --- Discover subsets ---
    if args.subsets:
        subset_names = args.subsets
    else:
        print("Auto-discovering subsets...")
        subset_names = discover_subsets(src_dir, mask_dir)

    if not subset_names:
        print("ERROR: No valid subsets found. Check src_dir and mask_dir paths.")
        return

    print(f"Source dir  : {src_dir}")
    print(f"Mask dir    : {mask_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Subsets     : {subset_names}")
    print(f"Verify masks: {verify_masks}")
    print(f"Workers     : {args.num_workers}")
    print()

    # --- Process each subset ---
    grand_eps = 0
    grand_steps = 0
    grand_missing = 0

    for subset_name in subset_names:
        print(f"[{subset_name}]")
        n_eps, n_steps, n_missing = process_subset(
            subset_name=subset_name,
            src_dir=src_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            num_workers=args.num_workers,
            start_ep=args.start_episode,
            end_ep=args.end_episode,
            verify_masks=verify_masks,
            copy_videos=args.copy_videos,
        )
        grand_eps += n_eps
        grand_steps += n_steps
        grand_missing += n_missing
        print()

    # --- Grand summary ---
    print("=" * 60)
    print(f"All done. {len(subset_names)} subsets, {grand_eps} episodes, {grand_steps} total steps.")
    if verify_masks:
        print(f"Missing mask files: {grand_missing}")
    if grand_missing > 0:
        print("WARNING: Some mask files are missing. Check mask generation output.")


if __name__ == "__main__":
    main()
