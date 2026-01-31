#!/usr/bin/env python3
"""
Run ACT policy on a new video: load video frames, build observation, run select_action, output actions.
Expects video resolution/fps to match training when possible; for multi-camera training,
pass multiple --video paths with matching camera keys (see --camera-keys).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

# Load .env before HF/lerobot
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

import cv2
from lerobot.common.policies.act.modeling_act import ACTPolicy


def load_video_frames(path: str, max_frames: int | None = None) -> list[torch.Tensor]:
    """Read video with OpenCV and return list of tensors (C, H, W) in [0, 1]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if max_frames is not None and len(frames) >= max_frames:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # (H, W, C) -> (C, H, W), float [0,1]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        frames.append(t)
    cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACT on a new video")
    parser.add_argument(
        "video",
        nargs="+",
        help="Path(s) to video file(s). One per camera; order must match --camera-keys.",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Local path or HF repo_id for ACT policy (e.g. lerobot/act_pusht or outputs/train/act_pusht)",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default="observation.images.top",
        help="Comma-separated observation.images keys (default: observation.images.top). Must match policy/dataset.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Optional path to state vector (one line of space-separated floats) or 'zeros' to use zeros.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame index to run inference on (-1 = last frame, default: -1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write predicted action (JSON or .pt) to this path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu", "mps"),
        help="Device for inference (default: cuda)",
    )
    args = parser.parse_args()

    if args.policy_path is None:
        parser.error("--policy-path is required (e.g. lerobot/act_pusht or path to checkpoint)")

    policy_path = args.policy_path
    camera_keys = [k.strip() for k in args.camera_keys.split(",")]
    videos = args.video
    if len(videos) != len(camera_keys):
        parser.error(
            f"Number of videos ({len(videos)}) must match number of camera keys ({len(camera_keys)})."
        )

    # Load policy
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.to(args.device)

    # Optionally get camera keys from policy config
    if hasattr(policy, "config") and hasattr(policy.config, "input_shapes"):
        # input_shapes may list observation.images.* keys
        pass  # use args.camera_keys unless we want to override

    # Load video(s) and pick frame; use flat keys like "observation.images.top"
    observation = {}
    for cam_key, video_path in zip(camera_keys, videos):
        frames = load_video_frames(video_path)
        if not frames:
            raise ValueError(f"No frames read from {video_path}")
        idx = args.frame_index if args.frame_index >= 0 else len(frames) + args.frame_index
        frame = frames[idx].to(args.device)
        # ACT may expect (1, C, H, W) or (C, H, W); add batch dim if needed
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        observation[cam_key] = frame

    # State: zeros or from file
    if args.state == "zeros" or args.state is None:
        # Infer state dim from policy if possible; else use a small default
        state_dim = getattr(
            getattr(policy, "config", None),
            "state_dim",
            None,
        )
        if state_dim is None and hasattr(policy, "config") and hasattr(policy.config, "input_shapes"):
            state_dim = policy.config.input_shapes.get("observation.state", [0])
            state_dim = state_dim[-1] if isinstance(state_dim, list) and state_dim else 2
        if state_dim is None:
            state_dim = 2  # pusht-like default
        if isinstance(state_dim, list):
            state_dim = state_dim[-1] if state_dim else 2
        state = torch.zeros(1, state_dim, device=args.device)
    else:
        with open(args.state) as f:
            line = f.readline()
        state = torch.tensor([[float(x) for x in line.split()]], device=args.device)

    observation["observation.state"] = state

    # select_action may expect batched observation; ensure batch dims
    with torch.no_grad():
        action = policy.select_action(observation)

    if isinstance(action, torch.Tensor):
        action = action.cpu()
        if action.dim() > 1:
            action = action.squeeze(0)
        action_list = action.tolist()
    else:
        action_list = action

    print("Predicted action:", action_list)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".json":
            with open(out_path, "w") as f:
                json.dump({"action": action_list}, f, indent=2)
        else:
            torch.save({"action": action}, out_path)
        print(f"Wrote to {args.output}")


if __name__ == "__main__":
    main()
