# SO101 Inference Streamlit App with Three.js 3D Viewer

## Summary

Adds a Streamlit app for running Diffusion policy inference on the SO101 dataset, with a Three.js 3D arm viewer, video playback, and correct frame alignment via LeRobot dataset API.

## What's Changed

### New / Modified Files

| File | Description |
|------|-------------|
| `scripts/app_inference_viz.py` | Streamlit app: LeRobot dataset or raw video input, inference, 3D viewer |
| `scripts/inference_utils.py` | Policy loading, `run_dataset_inference`, `run_video_inference`, `frames_to_video_bytes` |
| `scripts/so101_visualizer.py` | Pinocchio FK, trajectory for Three.js, gripper symmetrization |
| `scripts/so101_mujoco_viewer.html` | Three.js 3D arm viewer: Play/Pause, speed (fps), frame slider |
| `scripts/run_streamlit_app.sh` | Launch script (`streamlit run --server.address 0.0.0.0`) |
| `scripts/train_diffusion.py` | Add `--save-freq`, `--save_checkpoint=true` |

### Features

- **LeRobot dataset mode**: Select dataset root + episode index; uses correct front/overhead frame alignment (same episode, no chunk mismatch)
- **Raw video mode**: Fallback for custom front + overhead video paths
- **Max frames 100,000**: Process full video length; optional limit when unchecked
- **Video playback**: Front/overhead videos with native play controls; frames encoded to mp4 when using dataset
- **3D arm viewer (Three.js)**: Cylinders for links, spheres for joints, two-finger gripper; Play/Pause, adjustable speed (0.5–60 fps), frame slider
- **Pinocchio FK**: Uses SO101 URDF for forward kinematics; symmetrized gripper finger lengths
- **Batched inference**: `batch_size=32` for dataset mode

### Dependencies

- `streamlit`, `plotly`, `av`, `pin` (Pinocchio)

## Quick Start

```bash
# Run Streamlit app
./scripts/run_streamlit_app.sh
# or
uv run streamlit run scripts/app_inference_viz.py --server.address 0.0.0.0

# Configure: checkpoint path, LeRobot dataset root + episode index
# Click "Run Inference"
# View: videos, action curves, 3D arm (Play/Pause, adjust speed)
```

## Commits

- `391ddc1` feat: SO101 inference Streamlit app with Three.js 3D viewer
- `be06755` fix: inference processes full video length, not limited to 30 frames
- `66eb7eb` feat: LeRobot dataset inference, video playback, 3D viewer fixes
