#!/usr/bin/env python3
"""
Load a LeRobot dataset from ~/data and print basic info (num_episodes, num_frames, meta).
Optionally list ~/data subdirs that look like LeRobot datasets.
"""
from __future__ import annotations

import argparse
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

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


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
            print("Expected subdirs with data/ and meta/ (e.g. pusht).")
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

    episodes = None
    if args.episodes is not None:
        episodes = [int(x.strip()) for x in args.episodes.split(",")]

    try:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
    except TypeError:
        # Some versions use repo_id as full path for local datasets
        meta = LeRobotDatasetMetadata(os.path.join(root, repo_id))
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        try:
            ds = LeRobotDataset(repo_id, root=root, episodes=episodes or [0])
            meta = ds.meta
        except TypeError:
            ds = LeRobotDataset(os.path.join(root, repo_id), episodes=episodes or [0])
            meta = ds.meta
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            return

    print(f"Dataset: {repo_id} (root={root})")
    print(f"  total_episodes: {meta.total_episodes}")
    print(f"  total_frames:  {getattr(meta, 'total_frames', 'N/A')}")
    print(f"  fps:           {meta.fps}")
    print(f"  robot_type:    {getattr(meta, 'robot_type', 'N/A')}")
    print(f"  camera_keys:   {meta.camera_keys}")

    try:
        ds = LeRobotDataset(repo_id, root=root, episodes=episodes)
    except TypeError:
        ds = LeRobotDataset(os.path.join(root, repo_id), episodes=episodes)
    print(f"  loaded episodes: {ds.num_episodes}")
    print(f"  loaded frames:   {ds.num_frames}")
    if hasattr(ds.meta, "action_dim") or "action" in getattr(ds, "features", {}):
        print(f"  meta: {ds.meta}")


if __name__ == "__main__":
    main()
