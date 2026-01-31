#!/usr/bin/env python3
"""
Train ACT policy on a LeRobot dataset under ~/data.
Loads .env for HF token, then runs lerobot-train with dataset from local root.
If your lerobot version does not support --dataset.root, push the dataset to
Hugging Face Hub first or copy it to the cache dir; see README.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    # Load .env before any HF/lerobot usage
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

    parser = argparse.ArgumentParser(description="Train ACT on LeRobot dataset in ~/data")
    parser.add_argument(
        "dataset_name",
        help="Dataset name (subdir under DATA_ROOT, e.g. pusht)",
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/data"),
        help="Root directory containing the dataset (default: ~/data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory (default: outputs/train/act_<dataset_name>)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Job name for logging (default: act_<dataset_name>)",
    )
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="HF repo id for policy (optional; for push after training)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu", "mps"),
        help="Device for training (default: cuda)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the lerobot-train command, do not run",
    )
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dataset_name = args.dataset_name
    # lerobot expects dataset.root to be the directory that contains meta/, data/, videos/
    dataset_root = os.path.join(root, dataset_name)
    job_name = args.job_name or f"act_{dataset_name}"
    output_dir = args.output_dir or os.path.join("outputs", "train", job_name)
    policy_repo_id = args.policy_repo_id or f"act_{dataset_name}"

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={args.device}",
        f"--policy.repo_id={policy_repo_id}",
    ]

    if args.dry_run:
        print("Would run:")
        print(" ", " ".join(cmd))
        return

    # Do not create output_dir here; lerobot-train creates it and errors if it exists with resume=False
    ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if ret.returncode != 0:
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
