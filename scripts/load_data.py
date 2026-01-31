#!/usr/bin/env python3
"""
Load a LeRobot dataset from ~/data and print basic info (num_episodes, num_frames, meta).
Supports both v2.1 and v3.0 formats. Does not modify info.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Load .env before importing lerobot/huggingface_hub
try:
    from dotenv import load_dotenv
    load_dotenv()
    token = (
        os.environ.get("HF_access_token")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
except ImportError:
    pass

from lerobot.datasets.backward_compatibility import BackwardCompatibilityError
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def load_v21_metadata(dataset_root: Path) -> dict:
    """Load metadata from v2.1 format (info.json) without version check."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found at {dataset_root}")
    with open(info_path) as f:
        return json.load(f)


def print_dataset_info(repo_id: str, dataset_root: Path, info: dict, episodes: list[int] | None = None) -> None:
    """Print dataset info from loaded metadata."""
    total_episodes = info.get("total_episodes", "N/A")
    total_frames = info.get("total_frames", "N/A")
    fps = info.get("fps", "N/A")
    robot_type = info.get("robot_type", "N/A")
    features = info.get("features", {})
    camera_keys = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]

    print(f"Dataset: {repo_id} (root={dataset_root})")
    print(f"  codebase_version: {info.get('codebase_version', 'N/A')}")
    print(f"  total_episodes: {total_episodes}")
    print(f"  total_frames:  {total_frames}")
    print(f"  fps:           {fps}")
    print(f"  robot_type:    {robot_type}")
    print(f"  camera_keys:   {camera_keys}")
    if episodes:
        print(f"  requested episodes: {episodes}")


def find_lerobot_datasets(data_root: str) -> list[str]:
    """List subdirs of data_root that look like LeRobot v3 (have data/, videos/, meta/)."""
    root = Path(data_root).expanduser()
    if not root.is_dir():
        return []
    names = []
    for p in root.iterdir():
        if p.is_dir() and (p / "data").is_dir() and (p / "meta").is_dir():
            names.append(p.name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load LeRobot dataset from ~/data and print info")
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default=None,
        help="Dataset name (subdir under DATA_ROOT). If omitted, list available datasets.",
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/data"),
        help="Root directory containing dataset subdirs (default: ~/data)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices to load (e.g. 0,1,2). Default: all.",
    )
    args = parser.parse_args()
    root = os.path.expanduser(args.root)

    if args.dataset_name is None:
        names = find_lerobot_datasets(root)
        if not names:
            print(f"No LeRobot-style datasets found under {root}")
            print("Expected subdirs with data/ and meta/ (e.g. so101_bench_real_2_v2.1).")
            return
        print(f"Datasets under {root}:")
        for n in names:
            print(f"  {n}")
        return

    repo_id = args.dataset_name
    dataset_path = os.path.join(root, repo_id)
    if not os.path.isdir(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    # LeRobot expects root to be the dataset directory (containing meta/, data/, videos/).
    dataset_root = Path(dataset_path)

    episodes = None
    if args.episodes is not None:
        episodes = [int(x.strip()) for x in args.episodes.split(",")]

    try:
        meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
        print_dataset_info(repo_id, dataset_root, meta.info, episodes)
        print(f"  (loaded via LeRobot v3 API)")
        try:
            ds = LeRobotDataset(repo_id, root=dataset_root, episodes=episodes)
        except TypeError:
            ds = LeRobotDataset(str(dataset_root), episodes=episodes)
        print(f"  loaded episodes: {ds.num_episodes}")
        print(f"  loaded frames:   {ds.num_frames}")
    except BackwardCompatibilityError:
        # v2.1 format: load info.json directly, do not modify it
        info = load_v21_metadata(dataset_root)
        print_dataset_info(repo_id, dataset_root, info, episodes)
        print(f"  (v2.1 format - metadata loaded from info.json, no conversion applied)")
    except TypeError:
        meta = LeRobotDatasetMetadata(str(dataset_root))
        print_dataset_info(repo_id, dataset_root, meta.info, episodes)
        try:
            ds = LeRobotDataset(str(dataset_root), episodes=episodes)
            print(f"  loaded episodes: {ds.num_episodes}")
            print(f"  loaded frames:   {ds.num_frames}")
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to load: {e}")
        return


if __name__ == "__main__":
    main()
