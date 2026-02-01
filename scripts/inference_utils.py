#!/usr/bin/env python3
"""
Inference utilities for SO101 Diffusion/ACT policy: load policy, run video inference,
return per-frame actions. Uses LeRobot dataset (correct frame alignment) or raw videos.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
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

# Import config classes to register them with PreTrainedConfig before from_pretrained
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: F401
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

MAX_FRAMES = 100000


def frames_to_video_bytes(
    frames: list[torch.Tensor],
    fps: int = 30,
) -> bytes:
    """Encode frames (C,H,W) in [0,1] to mp4 bytes for st.video()."""
    if not frames:
        return b""
    import av
    buf = io.BytesIO()
    h, w = frames[0].shape[1], frames[0].shape[2]
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for fr in frames:
            arr = (fr.permute(1, 2, 0).numpy() * 255).astype("uint8")
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def resolve_checkpoint_path(path: str) -> str:
    """Resolve checkpoint path: add pretrained_model if config.json not found."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path) and not path.endswith("pretrained_model"):
            pretrained = os.path.join(path, "pretrained_model")
            if os.path.isdir(pretrained):
                return pretrained
    return path


def get_policy_class(policy_type: str):
    """Get policy class by type."""
    from lerobot.policies.factory import get_policy_class as _get
    return _get(policy_type)


def load_policy_and_processors(
    checkpoint_path: str,
    device: str = "cuda",
):
    """Load policy, preprocessor, and postprocessor from checkpoint."""
    path = resolve_checkpoint_path(checkpoint_path)
    config = PreTrainedConfig.from_pretrained(path)
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(path)
    policy.eval()
    policy.to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=path,
        config_filename=POLICY_PREPROCESSOR_DEFAULT_NAME + ".json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=path,
        config_filename=POLICY_POSTPROCESSOR_DEFAULT_NAME + ".json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor, config


def load_video_frames(
    source: str | Path | BinaryIO,
    max_frames: int | None = None,
) -> list[torch.Tensor]:
    """Read video from path or file-like; return list of tensors (C, H, W) in [0, 1]."""
    if hasattr(source, "read"):
        # BinaryIO: write to temp file for OpenCV
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(source.read())
            temp_path = f.name
        try:
            frames = _read_frames(temp_path, max_frames)
        finally:
            os.unlink(temp_path)
        return frames
    path = str(source)
    return _read_frames(path, max_frames)


def _read_frames(path: str, max_frames: int | None) -> list[torch.Tensor]:
    """Read video with pyav (supports AV1, etc.). Returns (C,H,W) tensors in [0,1]."""
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if max_frames is not None and len(frames) >= max_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        frames.append(t)
    container.close()
    return frames


def run_video_inference(
    checkpoint_path: str,
    front_video: str | Path | BinaryIO,
    overhead_video: str | Path | BinaryIO,
    device: str = "cuda",
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], list[torch.Tensor], list[torch.Tensor]]:
    """
    Run inference on front + overhead videos. Returns (actions, front_frames, overhead_frames).
    When max_frames is None, uses MAX_FRAMES (100000).
    """
    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    policy, preprocessor, postprocessor, config = load_policy_and_processors(
        checkpoint_path, device
    )

    front_frames = load_video_frames(front_video, max_frames=max_frames)
    overhead_frames = load_video_frames(overhead_video, max_frames=max_frames)
    n_frames = min(len(front_frames), len(overhead_frames))
    if n_frames == 0:
        raise ValueError("No frames in videos")

    image_keys = [k for k in config.input_features if "images" in k]
    state_ft = config.input_features.get("observation.state")
    state_dim = state_ft.shape[0] if state_ft and state_ft.shape else 6

    if hasattr(policy, "reset"):
        policy.reset()

    actions = []
    for i in range(n_frames):
        obs = {
            "observation.images.front": front_frames[i].unsqueeze(0).to(device),
            "observation.images.overhead": overhead_frames[i].unsqueeze(0).to(device),
            "observation.state": torch.zeros(1, state_dim, device=device),
        }
        obs = preprocessor(obs)
        with torch.no_grad():
            action = policy.select_action(obs)
        action = postprocessor(action)
        if isinstance(action, torch.Tensor):
            a = action.cpu().numpy()
        else:
            a = np.array(action)
        if a.ndim > 1:
            a = a.squeeze(0)
        actions.append(a)

    return actions, front_frames[:n_frames], overhead_frames[:n_frames]


def run_dataset_inference(
    checkpoint_path: str,
    dataset_root: str | Path,
    episode_index: int,
    device: str = "cuda",
    max_frames: int | None = None,
    batch_size: int = 32,
) -> tuple[list[np.ndarray], list[torch.Tensor], list[torch.Tensor]]:
    """
    Run inference on a LeRobot dataset episode. Uses correct video alignment
    (front and overhead frames are from the same episode).
    Returns (actions, front_frames, overhead_frames).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    max_frames = max_frames if max_frames is not None else MAX_FRAMES

    policy, preprocessor, postprocessor, config = load_policy_and_processors(
        checkpoint_path, device
    )

    dataset_root = Path(dataset_root).expanduser().resolve()
    repo_id = dataset_root.name
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root),
        episodes=[episode_index],
        video_backend="pyav",
    )

    n_frames = min(len(ds), max_frames)
    if n_frames == 0:
        raise ValueError(f"No frames in episode {episode_index}")

    state_ft = config.input_features.get("observation.state")
    state_dim = state_ft.shape[0] if state_ft and state_ft.shape else 6

    if hasattr(policy, "reset"):
        policy.reset()
    elif hasattr(policy, "module") and hasattr(policy.module, "reset"):
        policy.module.reset()

    actions = []
    front_frames = []
    overhead_frames = []

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        batch_front = []
        batch_overhead = []
        for i in range(start, end):
            item = ds[i]
            batch_front.append(item["observation.images.front"])
            batch_overhead.append(item["observation.images.overhead"])

        front_stack = torch.stack(batch_front).to(device)
        overhead_stack = torch.stack(batch_overhead).to(device)
        state_stack = torch.zeros(len(batch_front), state_dim, device=device)

        obs = {
            "observation.images.front": front_stack,
            "observation.images.overhead": overhead_stack,
            "observation.state": state_stack,
        }
        obs = preprocessor(obs)
        with torch.no_grad():
            action = policy.select_action(obs)
        action = postprocessor(action)

        if isinstance(action, torch.Tensor):
            a = action.cpu().numpy()
        else:
            a = np.array(action)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        for j in range(len(batch_front)):
            actions.append(a[j])
            front_frames.append(batch_front[j])
            overhead_frames.append(batch_overhead[j])

    return actions, front_frames, overhead_frames
