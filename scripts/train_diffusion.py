#!/usr/bin/env python3
"""
Train Diffusion policy on so101_bench_real dataset under ~/data.
Loads .env for HF token, then runs lerobot-train with policy.type=diffusion.

so101_bench_real contains video/image observations (observation.images.front,
observation.images.overhead) suitable for Diffusion Policy training.

Troubleshooting CUDA Error 802 (system not yet initialized):
  On 8x A100/H100 (NVSwitch) instances, NVIDIA Fabric Manager must be running.
  Install and start it:
    sudo apt-get install -y nvidia-fabricmanager-XXX   # XXX = driver version, e.g. 570 or 580
    sudo systemctl start nvidia-fabricmanager
  Verify: nvidia-smi -q | grep -A2 Fabric  (State should be "Completed")

Usage:
  uv run python scripts/train_diffusion.py so101_bench_real_2_v2.1
  uv run python scripts/train_diffusion.py so101_bench_real_2_v2.1 --root ~/data
  uv run python scripts/train_diffusion.py so101_bench_real_2_v2.1 --num-gpus 8   # DDP on all 8 GPUs
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
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for DDP (default: 1). Use 8 for all 8 A100s.",
    )
    parser.add_argument(
        "--mixed-precision",
        default="bf16",
        choices=("no", "fp16", "bf16"),
        help="Mixed precision for multi-GPU (default: bf16 for A100)",
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

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_args = cmd[1:]  # --dataset.repo_id=... etc

    if args.num_gpus > 1:
        # Use accelerate from same venv as sys.executable
        venv_bin = os.path.dirname(os.path.abspath(sys.executable))
        accelerate_bin = os.path.join(venv_bin, "accelerate")
        if not os.path.exists(accelerate_bin):
            accelerate_bin = "accelerate"
        run_cmd = [
            accelerate_bin,
            "launch",
            "--multi_gpu",
            f"--num_processes={args.num_gpus}",
        ]
        if args.mixed_precision != "no":
            run_cmd.append(f"--mixed_precision={args.mixed_precision}")
        run_cmd.extend(["-m", "lerobot.scripts.lerobot_train"])
        run_cmd.extend(train_args)
    else:
        run_cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train"] + train_args

    if args.dry_run:
        print("Would run:")
        print("  " + " ".join(run_cmd))
        return

    # Warm up CUDA driver before training (helps avoid Error 802: system not yet initialized)
    # On cloud instances (e.g. AWS), nvidia-smi can take 15+ seconds on first run
    if args.device == "cuda":
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=False,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    ret = subprocess.run(run_cmd, cwd=project_root)
    if ret.returncode != 0:
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
