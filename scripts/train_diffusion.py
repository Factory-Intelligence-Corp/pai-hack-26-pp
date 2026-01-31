#!/usr/bin/env python3
"""
Train Diffusion policy on so101_bench_real dataset under ~/data.
Loads .env for HF token, then runs lerobot-train with policy.type=diffusion.

so101_bench_real contains video/image observations (observation.images.front,
observation.images.overhead) suitable for Diffusion Policy training.

Usage:
  uv run python scripts/train_diffusion.py so101_bench_real_2_v2.1
  uv run python scripts/train_diffusion.py so101_bench_real_2_v2.1 --root ~/data
  uv run python scripts/train_diffusion.py so101_bench_real_2_v2.1 --output-dir outputs/train/diffusion_so101
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

    parser = argparse.ArgumentParser(
        description="Train Diffusion policy on LeRobot dataset in ~/data"
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default="so101_bench_real_2_v2.1",
        help="Dataset name (subdir under DATA_ROOT, default: so101_bench_real_2_v2.1)",
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/data"),
        help="Root directory containing the dataset (default: ~/data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory (default: outputs/train/diffusion_<dataset_name>)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Job name for logging (default: diffusion_<dataset_name>)",
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
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: policy preset, often 8 for diffusion)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Training steps (default: 100000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the lerobot-train command, do not run",
    )
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dataset_name = args.dataset_name
    dataset_root = os.path.join(root, dataset_name)
    job_name = args.job_name or f"diffusion_{dataset_name}"
    output_dir = args.output_dir or os.path.join("outputs", "train", job_name)
    policy_repo_id = args.policy_repo_id or f"diffusion_{dataset_name}"

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--policy.type=diffusion",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={args.device}",
        f"--policy.repo_id={policy_repo_id}",
    ]

    if args.batch_size is not None:
        cmd.append(f"--batch_size={args.batch_size}")
    if args.steps is not None:
        cmd.append(f"--steps={args.steps}")

    if args.dry_run:
        print("Would run:")
        print("  " + " ".join(cmd))
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ret = subprocess.run(cmd, cwd=project_root)
    if ret.returncode != 0:
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
