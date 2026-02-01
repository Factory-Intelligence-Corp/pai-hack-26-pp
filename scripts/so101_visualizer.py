#!/usr/bin/env python3
"""
3D visualization for SO101 arm motion from joint positions (degrees).
Uses Pinocchio + URDF for FK, Plotly for 3D display.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

SO101_URDF_URL = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
    "Simulation/SO101/so101_new_calib.urdf"
)
# Joint order in dataset/action: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
SO101_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper_joint",
]


def _get_urdf_path() -> Path:
    """Download URDF if needed, return local path."""
    cache_dir = Path.home() / ".cache" / "so101_urdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = cache_dir / "so101_new_calib.urdf"
    if not urdf_path.exists():
        urllib.request.urlretrieve(SO101_URDF_URL, urdf_path)
    return urdf_path


# SO101 kinematic chain frame names (in order, base to tip)
SO101_LINK_CHAIN = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
]
SO101_GRIPPER_FIXED = "gripper_frame_link"   # fixed finger tip
SO101_GRIPPER_MOVING = "moving_jaw_so101_v1_link"  # moving finger (jaw)


def _fk_pinocchio(joint_pos_deg: np.ndarray, urdf_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute link and gripper positions using Pinocchio FK.
    Returns (link_positions, gripper_positions).
    link_positions: 6 points (base through gripper_link).
    gripper_positions: [gripper_base, fixed_finger_tip, moving_finger_tip].
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    q = np.deg2rad(joint_pos_deg[:6].astype(float))
    if len(q) < model.nq:
        q = np.pad(q, (0, max(0, model.nq - len(q))))
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    link_positions = []
    for name in SO101_LINK_CHAIN:
        fid = model.getFrameId(name)
        if fid < model.nframes and hasattr(data, "oMf"):
            pos = np.array(data.oMf[fid].translation)
            link_positions.append(pos)
    if len(link_positions) < 2:
        links = _fk_simplified(joint_pos_deg)
        gripper = _gripper_simplified(joint_pos_deg, links[-1], links[-2] if len(links) > 1 else links[-1])
        return links, gripper

    gripper_base = link_positions[-1]
    gripper_positions = [gripper_base]
    for name in [SO101_GRIPPER_FIXED, SO101_GRIPPER_MOVING]:
        fid = model.getFrameId(name)
        if fid < model.nframes and hasattr(data, "oMf"):
            pos = np.array(data.oMf[fid].translation)
            gripper_positions.append(pos)
    if len(gripper_positions) < 3:
        gripper_positions = _gripper_simplified(
            joint_pos_deg, gripper_base, link_positions[-2] if len(link_positions) > 1 else gripper_base
        )
    return link_positions, gripper_positions


def _fk_simplified(joint_pos_deg: np.ndarray) -> list[np.ndarray]:
    """Simplified FK fallback: approximate link endpoints (meters)."""
    # Approximate SO101 link lengths (m), typical small arm
    L1, L2, L3, L4, L5 = 0.05, 0.15, 0.15, 0.08, 0.05
    q = np.deg2rad(joint_pos_deg[:6].astype(float))
    q = np.pad(q, (0, max(0, 6 - len(q))))

    def rotz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def roty(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    p0 = np.zeros(3)
    R = np.eye(3)
    R = R @ rotz(q[0])  # shoulder_pan
    p1 = p0 + R @ np.array([0, 0, L1])
    R = R @ roty(q[1])  # shoulder_lift
    p2 = p1 + R @ np.array([L2, 0, 0])
    R = R @ roty(q[2])  # elbow_flex
    p3 = p2 + R @ np.array([L3, 0, 0])
    R = R @ roty(q[3])  # wrist_flex
    p4 = p3 + R @ np.array([L4, 0, 0])
    R = R @ rotz(q[4])  # wrist_roll
    p5 = p4 + R @ np.array([L5, 0, 0])
    return [p0, p1, p2, p3, p4, p5]


def _gripper_simplified(
    joint_pos_deg: np.ndarray,
    gripper_base: np.ndarray | None = None,
    wrist_pos: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Simplified gripper: [base, fixed_finger_tip, moving_finger_tip]. Gripper joint (index 5) controls jaw open/close."""
    q = np.asarray(joint_pos_deg, dtype=float)[:6]
    q = np.pad(q, (0, 6 - len(q)))
    gripper_deg = q[5]
    if gripper_base is None or wrist_pos is None:
        links = _fk_simplified(joint_pos_deg)
        gripper_base = links[-1]
        wrist_pos = links[-2]
    forward = gripper_base - wrist_pos
    ln = np.linalg.norm(forward)
    forward = forward / ln if ln > 1e-6 else np.array([1, 0, 0])
    right = np.cross(forward, np.array([0, 0, 1]))
    right = right / (np.linalg.norm(right) + 1e-9)
    jaw_open = 0.02 * (gripper_deg / 90.0)  # 0–90 deg -> 0–2 cm opening
    fixed_tip = gripper_base + 0.03 * forward
    moving_tip = gripper_base - jaw_open * right + 0.02 * forward
    return [gripper_base, fixed_tip, moving_tip]


def build_threejs_viewer_html(trajectory: list[dict]) -> str:
    """Build full HTML for Three.js viewer with embedded trajectory."""
    import json
    html_path = Path(__file__).parent / "so101_mujoco_viewer.html"
    html = html_path.read_text()
    inj = f'<script>window.SO101_TRAJECTORY = {json.dumps(trajectory)};</script>'
    return inj + "\n" + html


def trajectory_for_threejs(actions: list[np.ndarray], urdf_path: Path | None = None) -> list[dict]:
    """
    Build trajectory for Three.js viewer: list of {links, gripper} per frame.
    links: list of [x,y,z] arrays; gripper: [base, fixed_tip, moving_tip].
    """
    traj = []
    for a in actions:
        link_pts, gripper_pts = joint_positions_3d(a, urdf_path)
        traj.append({
            "links": link_pts.tolist(),
            "gripper": [p.tolist() for p in gripper_pts],
        })
    return traj


def joint_positions_3d(joint_pos_deg: np.ndarray, urdf_path: Path | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Get 3D positions of links and gripper for visualization.
    Returns (link_pts, gripper_pts).
    link_pts: (N, 3) array of arm link positions (base through gripper_link).
    gripper_pts: [base, fixed_finger_tip, moving_finger_tip].
    Uses Pinocchio + URDF if available; otherwise simplified FK.
    """
    joint_pos_deg = np.asarray(joint_pos_deg, dtype=float)
    if joint_pos_deg.ndim == 0 or len(joint_pos_deg) < 6:
        joint_pos_deg = np.pad(
            np.atleast_1d(joint_pos_deg), (0, 6 - len(np.atleast_1d(joint_pos_deg)))
        )
    try:
        if urdf_path is None:
            urdf_path = _get_urdf_path()
        links, gripper = _fk_pinocchio(joint_pos_deg, urdf_path)
    except Exception:
        links = _fk_simplified(joint_pos_deg)
        gripper = _gripper_simplified(joint_pos_deg, links[-1], links[-2] if len(links) > 1 else links[-1])
    link_pts = np.array(links)
    if link_pts.ndim == 1:
        link_pts = link_pts.reshape(1, -1)
    return link_pts, gripper


def _compute_scene_bounds(all_links: list, all_grippers: list) -> dict:
    """Compute fixed axis ranges to prevent camera jitter during animation."""
    xs, ys, zs = [], [], []
    for pts in all_links:
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())
        zs.extend(pts[:, 2].tolist())
    for g in all_grippers:
        for p in g:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
    pad = 0.05
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    zmin, zmax = min(zs) - pad, max(zs) + pad
    return dict(
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        zaxis=dict(range=[zmin, zmax]),
    )


def render_arm_3d(
    joint_positions: np.ndarray,
    frame_index: int = 0,
    title: str | None = None,
) -> "plotly.graph_objects.Figure":
    """
    Create Plotly 3D figure for SO101 arm at given joint configuration.
    Draws arm links and gripper (two fingers with open/close).
    joint_positions: (6,) array in degrees.
    """
    import plotly.graph_objects as go

    link_pts, gripper_pts = joint_positions_3d(joint_positions)
    x, y, z = link_pts[:, 0], link_pts[:, 1], link_pts[:, 2]
    gb, gfx, gmv = gripper_pts[0], gripper_pts[1], gripper_pts[2]

    traces = []
    # Arm links
    traces.append(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines+markers",
            line=dict(color="rgb(70, 85, 105)", width=14),
            marker=dict(size=9, color="rgb(55, 65, 85)", line=dict(width=2, color="rgb(35, 42, 55)"), symbol="circle"),
            name="Arm",
        )
    )
    # Gripper: two fingers (fixed + moving jaw)
    traces.append(
        go.Scatter3d(
            x=[gb[0], gfx[0]], y=[gb[1], gfx[1]], z=[gb[2], gfx[2]],
            mode="lines+markers",
            line=dict(color="rgb(50, 130, 60)", width=10),
            marker=dict(size=6, color="rgb(40, 120, 55)"),
            name="Gripper (fixed)",
        )
    )
    traces.append(
        go.Scatter3d(
            x=[gb[0], gmv[0]], y=[gb[1], gmv[1]], z=[gb[2], gmv[2]],
            mode="lines+markers",
            line=dict(color="rgb(50, 130, 60)", width=10),
            marker=dict(size=6, color="rgb(40, 120, 55)"),
            name="Gripper (jaw)",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or f"SO101 Arm (frame {frame_index})",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data", bgcolor="rgb(248, 249, 251)",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


def render_arm_3d_animated(
    actions: list[np.ndarray],
    frame_duration_ms: int = 80,
) -> "plotly.graph_objects.Figure":
    """
    Create Plotly 3D figure with Play/Pause animation over time.
    Fixed camera/zoom to prevent jitter during playback.
    actions: list of (6,) arrays in degrees.
    """
    import plotly.graph_objects as go

    all_data = [joint_positions_3d(a) for a in actions]
    all_links = [d[0] for d in all_data]
    all_grippers = [d[1] for d in all_data]

    def make_frame_data(link_pts, gripper_pts):
        x, y, z = link_pts[:, 0], link_pts[:, 1], link_pts[:, 2]
        gb, gfx, gmv = gripper_pts[0], gripper_pts[1], gripper_pts[2]
        return [
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines+markers",
                line=dict(color="rgb(70, 85, 105)", width=14),
                marker=dict(size=9, color="rgb(55, 65, 85)", line=dict(width=2, color="rgb(35, 42, 55)"), symbol="circle"),
            ),
            go.Scatter3d(
                x=[gb[0], gfx[0]], y=[gb[1], gfx[1]], z=[gb[2], gfx[2]],
                mode="lines+markers",
                line=dict(color="rgb(50, 130, 60)", width=10),
                marker=dict(size=6, color="rgb(40, 120, 55)"),
            ),
            go.Scatter3d(
                x=[gb[0], gmv[0]], y=[gb[1], gmv[1]], z=[gb[2], gmv[2]],
                mode="lines+markers",
                line=dict(color="rgb(50, 130, 60)", width=10),
                marker=dict(size=6, color="rgb(40, 120, 55)"),
            ),
        ]

    fig = go.Figure(
        data=make_frame_data(all_links[0], all_grippers[0]),
        frames=[go.Frame(data=make_frame_data(lnk, grp), name=str(k)) for k, (lnk, grp) in enumerate(zip(all_links, all_grippers))],
    )

    # Fixed axis ranges to prevent camera jitter during animation
    scene_bounds = _compute_scene_bounds(all_links, all_grippers)
    fig.update_layout(
        title="SO101 Arm – Play to animate inference over time",
        scene=dict(
            xaxis=dict(title="X (m)", **scene_bounds["xaxis"]),
            yaxis=dict(title="Y (m)", **scene_bounds["yaxis"]),
            zaxis=dict(title="Z (m)", **scene_bounds["zaxis"]),
            aspectmode="data",
            bgcolor="rgb(248, 249, 251)",
            uirevision="fixed",  # Preserve camera on frame change
        ),
        margin=dict(l=0, r=0, b=60, t=50),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.1, xanchor="left", y=0, yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0, x=0.1, len=0.9, xanchor="left", y=0, yanchor="top", pad=dict(t=50, b=10),
                currentvalue=dict(visible=True, prefix="Frame: ", xanchor="center"),
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0), mode="immediate")], label=str(i), method="animate") for i, f in enumerate(fig.frames)],
            )
        ],
    )
    return fig
