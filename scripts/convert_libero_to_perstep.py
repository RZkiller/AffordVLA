"""
Convert LeRobot-format LIBERO datasets (parquet + video) to DreamVLA per-step
offline format (directory tree + H5 + JPEG).

The output is aligned frame-by-frame with affordvla's LeRobotSingleDataset so that
per-step index N corresponds exactly to parquet row N (base_index=N, delta=0).

Usage:
    # Single dataset
    python scripts/convert_libero_to_perstep.py \
        --src_dir /path/to/libero_goal_no_noops_1.0.0_lerobot \
        --tgt_dir /path/to/libero_goal_converted \
        --dataset_name libero_goal \
        --num_workers 8

    # All four LIBERO subsets
    for ds in libero_object libero_goal libero_spatial libero_10; do
        python scripts/convert_libero_to_perstep.py \
            --src_dir /path/to/LEROBOT_LIBERO_DATA/${ds}_no_noops_1.0.0_lerobot \
            --tgt_dir /path/to/output/${ds}_converted \
            --dataset_name $ds
    done

Output structure (identical to DreamVLA convert_libero_per_step.py):
    {tgt_dir}/
    ├── meta_info.h5                       # num_episodes
    └── episodes/
        └── {episode_id:06d}/
            ├── meta_info.h5               # length
            └── steps/
                └── {step_id:04d}/
                    ├── other.h5           # action, observation/*, language_instruction, ...
                    ├── image_primary.jpg
                    └── image_wrist.jpg
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Video decoding – try multiple backends, same logic as affordvla video.py
# ---------------------------------------------------------------------------
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


def decode_all_frames_decord(video_path: str, timestamps: np.ndarray) -> np.ndarray:
    """Decode frames at given timestamps (seconds) using decord."""
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)
    frame_ts = vr.get_frame_timestamp(range(num_frames))
    indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
    return vr.get_batch(indices).asnumpy()


def decode_all_frames_pyav(video_path: str, timestamps: np.ndarray) -> np.ndarray:
    """Decode frames at given timestamps (seconds) using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)
    fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)

    # Decode ALL frames once (more efficient than seeking per-frame for full episode)
    all_frames = []
    all_pts = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_ndarray(format="rgb24"))
        all_pts.append(float(frame.pts * time_base))
    container.close()

    all_pts = np.array(all_pts)
    # Map timestamps to closest decoded frames
    indices = np.abs(all_pts[:, None] - timestamps[None, :]).argmin(axis=0)
    return np.array([all_frames[i] for i in indices])


def decode_all_frames(video_path: str, timestamps: np.ndarray, backend: str = "pyav") -> np.ndarray:
    """Decode frames at given timestamps using the specified backend."""
    if backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not installed. Install it or use --video_backend pyav")
        return decode_all_frames_decord(video_path, timestamps)
    elif backend == "pyav":
        if not PYAV_AVAILABLE:
            raise ImportError("PyAV is not installed. Install it or use --video_backend decord")
        return decode_all_frames_pyav(video_path, timestamps)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


# ---------------------------------------------------------------------------
# LeRobot metadata helpers
# ---------------------------------------------------------------------------

def load_info(src_dir: Path) -> dict:
    with open(src_dir / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes(src_dir: Path) -> list[dict]:
    with open(src_dir / "meta" / "episodes.jsonl") as f:
        return [json.loads(line) for line in f]


def load_tasks(src_dir: Path) -> dict[int, str]:
    """Return {task_index: task_description}."""
    with open(src_dir / "meta" / "tasks.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return {r["task_index"]: r["task"] for r in rows}


# ---------------------------------------------------------------------------
# Per-episode conversion worker
# ---------------------------------------------------------------------------

def convert_episode(args: tuple) -> tuple[str, int]:
    """Convert a single episode.  Returns (episode_id_str, num_steps)."""
    (
        src_dir,
        tgt_dir,
        episode_index,
        episode_length,
        data_path_pattern,
        video_path_pattern,
        chunk_size,
        tasks_map,
        video_backend,
    ) = args

    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    episode_id_str = str(episode_index).zfill(6)
    chunk_index = episode_index // chunk_size

    # --- Load parquet ---
    parquet_path = src_dir / data_path_pattern.format(
        episode_chunk=chunk_index, episode_index=episode_index
    )
    df = pd.read_parquet(parquet_path)
    num_steps = len(df)

    # --- Actions (7D) ---
    actions = np.stack(df["action"].values)  # (T, 7)

    # --- State = observation.state (8D: x,y,z,roll,pitch,yaw,pad,gripper) ---
    states = np.stack(df["observation.state"].values)  # (T, 8)

    # --- Timestamps ---
    timestamps = df["timestamp"].to_numpy()  # (T,)

    # --- Language instruction (from task_index column) ---
    task_idx = df["task_index"].iloc[0]
    if isinstance(task_idx, (np.integer, np.floating)):
        task_idx = int(task_idx)
    language_instruction = tasks_map[task_idx]

    # --- Gripper state (DreamVLA convention: shifted by 1 step) ---
    # gripper_state[t] = action[t-1, -1], gripper_state[0] = action[0, -1]
    gripper_state = np.zeros(num_steps)
    gripper_state[1:] = actions[:-1, -1]
    gripper_state[0] = actions[0, -1]

    # --- Decode video frames ---
    # Primary camera
    primary_video_path = src_dir / video_path_pattern.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key="observation.images.image",
    )
    try:
        primary_frames = decode_all_frames(str(primary_video_path), timestamps, video_backend)
    except Exception as e:
        raise RuntimeError(
            f"Primary decode failed. episode={episode_id_str}, path={primary_video_path}, backend={video_backend}"
        ) from e
    
    # Wrist camera
    wrist_video_path = src_dir / video_path_pattern.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key="observation.images.wrist_image",
    )
    try:
        wrist_frames = decode_all_frames(str(wrist_video_path), timestamps, video_backend)
    except Exception:
        print(f"Wrist decode failed. episode={episode_id_str}, path={wrist_video_path}, backend={video_backend}")
        wrist_frames = primary_frames

    # --- Create output directories ---
    episode_dir = tgt_dir / "episodes" / episode_id_str
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Episode meta_info.h5
    with h5py.File(str(episode_dir / "meta_info.h5"), "w") as f:
        f.create_dataset("length", data=num_steps)

    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(exist_ok=True)

    for step_idx in range(num_steps):
        step_dir = steps_dir / str(step_idx).zfill(4)
        step_dir.mkdir(exist_ok=True)

        # Save images
        Image.fromarray(primary_frames[step_idx]).save(str(step_dir / "image_primary.jpg"))
        Image.fromarray(wrist_frames[step_idx]).save(str(step_dir / "image_wrist.jpg"))

        # Save other.h5
        with h5py.File(str(step_dir / "other.h5"), "w") as h5f:
            # Language instruction
            h5f.create_dataset(
                "language_instruction",
                data=np.array(language_instruction, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            # Episode length
            h5f.create_dataset("episode_length", data=num_steps)
            # Action (7D)
            h5f.create_dataset("action", data=actions[step_idx])

            # Observation group
            obs_grp = h5f.create_group("observation")
            # proprio – EE state (8D: x,y,z,roll,pitch,yaw,pad,gripper)
            obs_grp.create_dataset("proprio", data=states[step_idx])
            # tcp_pose – first 6 dims of state (x,y,z,roll,pitch,yaw)
            obs_grp.create_dataset("tcp_pose", data=states[step_idx, :6])
            # gripper_state (scalar, shifted by 1 step per DreamVLA convention)
            obs_grp.create_dataset("gripper_state", data=gripper_state[step_idx])
            # gripper_position (from state's gripper dimension)
            obs_grp.create_dataset("gripper_position", data=states[step_idx, 7:8])

    return episode_id_str, num_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot LIBERO dataset to DreamVLA per-step format"
    )
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Root of a single LeRobot dataset (e.g. libero_goal_no_noops_1.0.0_lerobot)")
    parser.add_argument("--tgt_dir", type=str, required=True,
                        help="Output directory for per-step data")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name for data_info json (e.g. libero_goal)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--video_backend", type=str, default="pyav",
                        choices=["pyav", "decord"],
                        help="Video decoding backend")
    parser.add_argument("--start_episode", type=int, default=None,
                        help="First episode index to convert (inclusive)")
    parser.add_argument("--end_episode", type=int, default=None,
                        help="Last episode index to convert (exclusive)")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # --- Load metadata ---
    info = load_info(src_dir)
    episodes = load_episodes(src_dir)
    tasks_map = load_tasks(src_dir)

    data_path_pattern = info["data_path"]       # e.g. "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_path_pattern = info["video_path"]     # e.g. "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    chunk_size = info["chunks_size"]
    total_episodes = len(episodes)

    start_ep = args.start_episode if args.start_episode is not None else 0
    end_ep = args.end_episode if args.end_episode is not None else total_episodes

    print(f"Source: {src_dir}")
    print(f"Target: {tgt_dir}")
    print(f"Episodes: {start_ep} → {end_ep} (total {total_episodes})")
    print(f"Video backend: {args.video_backend}")
    print(f"Workers: {args.num_workers}")

    # --- Prepare worker arguments ---
    worker_args = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]
        if ep_idx < start_ep or ep_idx >= end_ep:
            continue
        worker_args.append((
            str(src_dir),
            str(tgt_dir),
            ep_idx,
            ep_len,
            data_path_pattern,
            video_path_pattern,
            chunk_size,
            tasks_map,
            args.video_backend,
        ))

    # --- Run conversion ---
    data_info = []
    if args.num_workers <= 1:
        for wa in worker_args:
            ep_id_str, n_steps = convert_episode(wa)
            data_info.append([ep_id_str, n_steps])
            print(f"  Episode {ep_id_str}: {n_steps} steps")
    else:
        with Pool(processes=args.num_workers) as pool:
            for ep_id_str, n_steps in pool.imap_unordered(convert_episode, worker_args):
                data_info.append([ep_id_str, n_steps])
                print(f"  Episode {ep_id_str}: {n_steps} steps")

    # Sort by episode ID for deterministic output
    data_info.sort(key=lambda x: x[0])

    # --- Write global meta_info.h5 ---
    with h5py.File(str(tgt_dir / "meta_info.h5"), "w") as f:
        f.create_dataset("num_episodes", data=len(data_info))

    # --- Write data_info json ---
    data_info_dir = Path("data_info")
    data_info_dir.mkdir(exist_ok=True)
    data_info_path = data_info_dir / f"{args.dataset_name}_converted.json"
    with open(data_info_path, "w") as f:
        json.dump(data_info, f)
    print(f"\ndata_info saved to {data_info_path}")

    total_steps = sum(n for _, n in data_info)
    print(f"Done. {len(data_info)} episodes, {total_steps} total steps.")


if __name__ == "__main__":
    main()
