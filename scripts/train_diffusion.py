#!/usr/bin/env python3
"""
Train Diffusion policy on so101_bench_real dataset under ~/data.
Loads .env for HF token, then runs lerobot-train with policy.type=diffusion.

so101_bench_real contains video/image observations (observation.images.front,
observation.images.overhead) suitable for Diffusion Policy training.

Troubleshooting:
  - ChildFailedError exitcode 1: Run with --log-file train.log to capture full error.
    Try --num-gpus 1 first; if single-GPU works, reduce --num-workers or --batch-size for multi-GPU.
  - SignalException/signal 15: Process was killed (Ctrl+C, OOM, or session disconnect).
    No checkpoints if killed before first save (checkpoints saved periodically in output_dir).
  - CUDA Error 802 (system not yet initialized):
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (enabled by default when WANDB_API_KEY in .env)",
    )
    parser.add_argument(
        "--wandb-project",
        default="PAI_hackathon",
        help="Wandb project name (default: PAI_hackathon)",
    )
    parser.add_argument(
        "--wandb-entity",
        default="littledesk",
        help="Wandb entity/team (default: littledesk)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 2 for multi-GPU, 4 for single-GPU)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write output to file (for debugging failures)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (requires --config-path)",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to train_config.json for resume (e.g. outputs/train/.../checkpoints/last/pretrained_model/train_config.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir if it exists and start fresh",
    )
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dataset_name = args.dataset_name
    dataset_root = os.path.join(root, dataset_name)
    job_name = args.job_name or f"diffusion_{dataset_name}"
    output_dir = args.output_dir or os.path.join("outputs", "train", job_name)
    policy_repo_id = args.policy_repo_id or f"diffusion_{dataset_name}"

    # Check dataset exists with LeRobot structure (meta/info.json)
    meta_info = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.exists(meta_info):
        print(
            f"Error: Dataset not found or invalid. Expected {meta_info}\n"
            f"  Ensure --root points to a directory containing LeRobot datasets (with meta/, data/, videos/).\n"
            f"  Run: python scripts/load_data.py  # to list available datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    # Overwrite: remove stale output dir from failed runs
    if args.overwrite and os.path.isdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"Removed existing {output_dir} (--overwrite)")

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
    nw = args.num_workers
    if nw is None:
        nw = 2 if args.num_gpus > 1 else 4  # reduce workers for multi-GPU to avoid fd/mem pressure
    cmd.append(f"--num_workers={nw}")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.resume:
        cmd.append("--resume=true")
        if not args.config_path:
            print(
                "Error: --resume requires --config-path pointing to train_config.json\n"
                "  Example: --config-path outputs/train/diffusion_xxx/checkpoints/last/pretrained_model/train_config.json",
                file=sys.stderr,
            )
            sys.exit(1)
        # Use absolute path so LeRobot treats it as local file, not HF repo
        config_path_abs = os.path.normpath(os.path.join(project_root, args.config_path))
        if not os.path.isfile(config_path_abs):
            print(f"Error: config_path not found: {config_path_abs}", file=sys.stderr)
            sys.exit(1)
        cmd.append(f"--config_path={config_path_abs}")
    if args.no_wandb:
        cmd.append("--wandb.enable=false")
    else:
        cmd.append("--wandb.enable=true")
        cmd.append(f"--wandb.project={args.wandb_project}")
        cmd.append(f"--wandb.entity={args.wandb_entity}")

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

    if args.log_file:
        # Capture full output for debugging (tee to file)
        log_path = os.path.join(project_root, args.log_file)
        proc = subprocess.Popen(
            run_cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with open(log_path, "w") as f:
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
                f.flush()
        proc.wait()
        ret = subprocess.CompletedProcess(run_cmd, proc.returncode)
    else:
        ret = subprocess.run(run_cmd, cwd=project_root)
    if ret.returncode != 0:
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
