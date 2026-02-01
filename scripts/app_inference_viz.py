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

from scripts.inference_utils import run_video_inference
from scripts.so101_visualizer import build_threejs_viewer_html, trajectory_for_threejs

DEFAULT_CHECKPOINT = (
    "outputs/train/diffusion_so101_bench_real_2_v2.1/checkpoints/004000/pretrained_model"
)
DEFAULT_FRONT = "~/datasets/5hadytru/so101_bench_real_2_v2.1/videos/observation.images.front/chunk-000/file-000.mp4"
DEFAULT_OVERHEAD = "~/datasets/5hadytru/so101_bench_real_2_v2.1/videos/observation.images.overhead/chunk-000/file-000.mp4"

st.set_page_config(
    page_title="SO101 Inference & 3D Viewer",
    page_icon="🤖",
    layout="wide",
)

st.title("SO101 Diffusion Policy Inference & 3D Arm Viewer")
st.markdown(
    "Upload or specify server paths for **front** and **overhead** videos "
    "(format: 480x640, 30fps, same as so101_bench_real_2_v2.1), run inference, "
    "and visualize the predicted arm motion in 3D."
)

with st.sidebar:
    st.header("Configuration")
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=DEFAULT_CHECKPOINT,
        help="Path to pretrained_model (e.g. .../checkpoints/004000/pretrained_model)",
    )
    process_all = st.checkbox(
        "Process all frames (full video length)",
        value=True,
        help="When unchecked, limit to max frames below",
    )
    max_frames = st.number_input(
        "Max frames (when not processing all)",
        min_value=1,
        max_value=100000,
        value=1000,
        help="Only used when 'Process all frames' is unchecked",
        disabled=process_all,
    )

    st.subheader("Video Input (front + overhead)")
    input_mode = st.radio("Input mode", ["Server path", "Upload"], horizontal=True)

    front_path = None
    overhead_path = None
    front_upload = None
    overhead_upload = None

    if input_mode == "Server path":
        front_path = st.text_input(
            "Front video path",
            value=os.path.expanduser(DEFAULT_FRONT),
            help="e.g. ~/datasets/.../observation.images.front/.../file-000.mp4",
        )
        overhead_path = st.text_input(
            "Overhead video path",
            value=os.path.expanduser(DEFAULT_OVERHEAD),
            help="e.g. ~/datasets/.../observation.images.overhead/.../file-000.mp4",
        )
    else:
        front_upload = st.file_uploader("Front video (mp4)", type=["mp4"])
        overhead_upload = st.file_uploader("Overhead video (mp4)", type=["mp4"])

    run_btn = st.button("Run Inference")

def has_video_input():
    if input_mode == "Server path":
        fp = os.path.expanduser(front_path or "")
        op = os.path.expanduser(overhead_path or "")
        return bool(fp and op and os.path.isfile(fp) and os.path.isfile(op))
    return bool(front_upload and overhead_upload)

if run_btn:
    if not checkpoint_path:
        st.error("Please specify a checkpoint path.")
    elif not has_video_input():
        st.error("Please provide both front and overhead videos (path or upload).")
    else:
        with st.spinner("Running inference..."):
            try:
                if input_mode == "Server path":
                    front = os.path.expanduser(front_path)
                    overhead = os.path.expanduser(overhead_path)
                    front_video_src = front
                    overhead_video_src = overhead
                else:
                    front_bytes = front_upload.read()
                    overhead_bytes = overhead_upload.read()
                    front = io.BytesIO(front_bytes)
                    overhead = io.BytesIO(overhead_bytes)
                    front_video_src = front_bytes
                    overhead_video_src = overhead_bytes
                actions, front_frames, overhead_frames = run_video_inference(
                    checkpoint_path,
                    front,
                    overhead,
                    device="cuda",
                    max_frames=None if process_all else int(max_frames),
                )
                st.session_state["actions"] = actions
                st.session_state["front_frames"] = front_frames
                st.session_state["overhead_frames"] = overhead_frames
                st.session_state["front_video_src"] = front_video_src
                st.session_state["overhead_video_src"] = overhead_video_src
                n_front = len(front_frames)
                n_overhead = len(overhead_frames)
                msg = f"Inferred {len(actions)} actions from {n_front} front + {n_overhead} overhead frames."
                if n_front != n_overhead:
                    msg += f" (Used min={len(actions)} frames — different video lengths may be from different recording durations or chunking.)"
                st.success(msg)
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
