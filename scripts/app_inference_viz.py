#!/usr/bin/env python3
"""
Streamlit app for SO101 Diffusion policy inference and 3D visualization.
Upload or specify server paths for front + overhead videos, run inference,
and visualize the predicted arm motion in 3D.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

# Register diffusion config before loading
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: F401
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401

import streamlit.components.v1 as components

from scripts.inference_utils import frames_to_video_bytes, run_dataset_inference, run_video_inference
from scripts.so101_visualizer import build_threejs_viewer_html, trajectory_for_threejs

DEFAULT_CHECKPOINT = (
    "outputs/train/diffusion_so101_bench_real_2_v2.1/checkpoints/004000/pretrained_model"
)
DEFAULT_DATASET_ROOT = "~/datasets/5hadytru/so101_bench_real_2_v2.1"

st.set_page_config(
    page_title="SO101 Inference & 3D Viewer",
    page_icon="🤖",
    layout="wide",
)

st.title("SO101 Diffusion Policy Inference & 3D Arm Viewer")
st.markdown(
    "Use **LeRobot dataset** (correct frame alignment) or raw videos. "
    "Max frames 100,000. Use LeRobot dataset for correct front/overhead alignment."
)

with st.sidebar:
    st.header("Configuration")
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=DEFAULT_CHECKPOINT,
        help="Path to pretrained_model",
    )
    input_mode = st.radio("Input mode", ["LeRobot dataset", "Raw videos"], horizontal=True)

    if input_mode == "LeRobot dataset":
        dataset_root = st.text_input(
            "Dataset root",
            value=os.path.expanduser(DEFAULT_DATASET_ROOT),
            help="e.g. ~/datasets/5hadytru/so101_bench_real_2_v2.1",
        )
        episode_index = st.number_input(
            "Episode index",
            min_value=0,
            max_value=100000,
            value=0,
            help="Episode to run inference on",
        )
    else:
        front_path = st.text_input(
            "Front video path",
            value=os.path.expanduser(DEFAULT_DATASET_ROOT + "/videos/observation.images.front/chunk-000/file-000.mp4"),
        )
        overhead_path = st.text_input(
            "Overhead video path",
            value=os.path.expanduser(DEFAULT_DATASET_ROOT + "/videos/observation.images.overhead/chunk-000/file-000.mp4"),
        )

    process_all = st.checkbox(
        "Process all frames",
        value=True,
        help="When unchecked, limit to max frames",
    )
    max_frames = st.number_input(
        "Max frames (when not all)",
        min_value=1,
        max_value=100000,
        value=1000,
        disabled=process_all,
    )

    run_btn = st.button("Run Inference")

def has_dataset_input():
    if input_mode != "LeRobot dataset":
        return False
    root = os.path.expanduser(dataset_root or "")
    meta = os.path.join(root, "meta", "info.json")
    return bool(root and os.path.isfile(meta))

def has_raw_input():
    if input_mode != "Raw videos":
        return False
    fp = os.path.expanduser(front_path or "")
    op = os.path.expanduser(overhead_path or "")
    return bool(fp and op and os.path.isfile(fp) and os.path.isfile(op))

if run_btn:
    if not checkpoint_path:
        st.error("Please specify a checkpoint path.")
    elif input_mode == "LeRobot dataset" and not has_dataset_input():
        st.error("Please specify a valid dataset root (with meta/info.json).")
    elif input_mode == "Raw videos" and not has_raw_input():
        st.error("Please specify valid front and overhead video paths.")
    else:
        with st.spinner("Running inference..."):
            try:
                max_n = None if process_all else int(max_frames)
                if input_mode == "LeRobot dataset":
                    root = os.path.expanduser(dataset_root)
                    actions, front_frames, overhead_frames = run_dataset_inference(
                        checkpoint_path,
                        root,
                        int(episode_index),
                        device="cuda",
                        max_frames=max_n,
                    )
                    # Encode frames to video bytes for play button (dataset has no raw video file)
                    front_video_src = frames_to_video_bytes(front_frames)
                    overhead_video_src = frames_to_video_bytes(overhead_frames)
                else:
                    front = os.path.expanduser(front_path)
                    overhead = os.path.expanduser(overhead_path)
                    front_video_src, overhead_video_src = front, overhead
                    actions, front_frames, overhead_frames = run_video_inference(
                        checkpoint_path,
                        front,
                        overhead,
                        device="cuda",
                        max_frames=max_n,
                    )
                st.session_state["actions"] = actions
                st.session_state["front_frames"] = front_frames
                st.session_state["overhead_frames"] = overhead_frames
                st.session_state["front_video_src"] = front_video_src
                st.session_state["overhead_video_src"] = overhead_video_src
                src = f"episode {episode_index}" if input_mode == "LeRobot dataset" else "raw videos"
                st.success(f"Inferred {len(actions)} actions from {src}.")
            except Exception as e:
                st.error(f"Inference failed: {e}")
                import traceback
                st.code(traceback.format_exc())

if "actions" in st.session_state:
    actions = st.session_state["actions"]
    front_frames = st.session_state["front_frames"]
    overhead_frames = st.session_state["overhead_frames"]
    front_video_src = st.session_state.get("front_video_src")
    overhead_video_src = st.session_state.get("overhead_video_src")
    n_frames = len(actions)

    frame_idx = st.slider("Frame", 0, n_frames - 1, 0, key="frame_slider")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Front camera")
        if front_video_src is not None:
            st.video(front_video_src)
        elif frame_idx < len(front_frames):
            fr = front_frames[frame_idx]
            img = (fr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            st.image(img, use_container_width=True)
    with col2:
        st.subheader("Overhead camera")
        if overhead_video_src is not None:
            st.video(overhead_video_src)
        elif frame_idx < len(overhead_frames):
            fr = overhead_frames[frame_idx]
            img = (fr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            st.image(img, use_container_width=True)

    st.subheader("Action (joint angles, degrees)")
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    action_at_frame = actions[frame_idx]
    st.write(dict(zip(joint_names, [round(float(x), 2) for x in action_at_frame])))

    st.subheader("Action curve (all frames)")
    actions_arr = np.array(actions)
    import plotly.graph_objects as go
    fig_curve = go.Figure()
    for i, name in enumerate(joint_names):
        fig_curve.add_trace(go.Scatter(y=actions_arr[:, i], name=name))
    fig_curve.add_vline(x=frame_idx, line_dash="dash", line_color="gray")
    fig_curve.update_layout(height=250, xaxis_title="Frame", yaxis_title="Angle (deg)")
    st.plotly_chart(fig_curve, use_container_width=True)

    st.subheader("3D Arm viewer (Three.js)")
    traj = trajectory_for_threejs(actions)
    viewer_html = build_threejs_viewer_html(traj)
    components.html(viewer_html, height=500, scrolling=False)
    st.caption("Play/Pause to animate. Use frame slider or drag to rotate view.")
else:
    st.info("Configure checkpoint and videos, then click **Run Inference**.")
