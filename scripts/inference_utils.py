#!/usr/bin/env python3
"""
Inference utilities for SO101 Diffusion/ACT policy: load policy, run video inference,
return per-frame actions. Supports server paths and uploaded videos.
"""
from __future__ import annotations

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
    actions: list of 6-dim numpy arrays (degrees).
    """
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
