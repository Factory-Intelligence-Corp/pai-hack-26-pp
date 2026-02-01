# Quick Start: Offline Latent Imitation

This guide will get you up and running with offline latent imitation in 5 minutes.

## What You'll Need

1. A demonstration video (MP4, AVI, etc.)
2. A trained policy with LAM conditioning
3. villa-x LAM model (automatically downloaded)

## Installation

```bash
# Clone repository with submodules
git clone --recursive https://github.com/your-repo/pai-hack-26-pp.git
cd pai-hack-26-pp

# Install dependencies
pip install torch torchvision opencv-python einops tqdm
pip install lerobot  # or install from source
```

## Minimal Example

Here's the simplest way to extract latent actions from a video:

```python
import sys
from pathlib import Path
import torch

# Add villa-x to path
sys.path.insert(0, str(Path(__file__).parents[1] / "villa-x"))

from lam import IgorModel
from lerobot.examples.offline_latent_imitation_robot import LatentActionExtractor

# 1. Create extractor
extractor = LatentActionExtractor(
    lam_model_path="microsoft/villa-x",
    device="cuda"
)

# 2. Load video and extract latent actions
frames = extractor.load_video("demo.mp4")
latent_actions = extractor.extract_from_frames(frames)

# 3. Save for later use
torch.save({"latent_actions": latent_actions}, "latent_actions.pt")

print(f"Extracted {len(latent_actions)} latent actions!")
print(f"Shape: {latent_actions.shape}")
```

## Command Line Usage

### Step 1: Extract Latent Actions

```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo_video.mp4 \
    --policy-path outputs/checkpoint.pt \
    --mode extract_only \
    --output-dir ./latents/
```

**Output:**
- `./latents/latent_actions.pt` - Saved latent actions

### Step 2: Run Inference (Simulation)

```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo_video.mp4 \
    --policy-path outputs/pusht_lam/checkpoint_epoch_100.pt \
    --mode simulation \
    --env-name PushT-v0 \
    --output-dir ./inference_results/
```

**Output:**
- `./inference_results/latent_actions.pt` - Latent actions
- `./inference_results/inference_simulation.mp4` - Recorded inference video

## Complete Workflow

### 1. Train a Policy with LAM Conditioning

First, precompute LAM tokens for your dataset:

```bash
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --lam-model-path microsoft/villa-x \
    --camera-key observation.images.top \
    --output-dir ./data/pusht_with_lam
```

Then train the policy:

```bash
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id lerobot/pusht \
    --lam-tokens-path ./data/pusht_with_lam/lam_tokens.pt \
    --output-dir ./outputs/pusht_lam \
    --num-epochs 100
```

### 2. Record or Obtain a Demonstration Video

The video should show the desired behavior you want to imitate. It can be:
- Screen recording of simulation
- Camera recording of real robot
- Any video showing the task

**Tips:**
- Use clear, well-lit videos
- Keep the camera angle consistent
- Match the FPS to your policy's expected rate

### 3. Extract Latent Actions

```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path my_demo.mp4 \
    --policy-path ./outputs/pusht_lam/checkpoint_epoch_100.pt \
    --mode extract_only
```

### 4. Run Inference

For simulation:
```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path my_demo.mp4 \
    --policy-path ./outputs/pusht_lam/checkpoint_epoch_100.pt \
    --mode simulation \
    --env-name PushT-v0
```

For real robot (requires robot config):
```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path my_demo.mp4 \
    --policy-path ./outputs/robot_lam/checkpoint.pt \
    --mode robot \
    --robot-config configs/my_robot.yaml
```

## Python API Usage

For programmatic usage:

```python
from pathlib import Path
import torch
from lerobot.examples.offline_latent_imitation_robot import (
    LatentActionExtractor,
    LatentGuidedInference
)

# Extract latent actions
extractor = LatentActionExtractor(
    lam_model_path="microsoft/villa-x",
    device="cuda"
)
frames = extractor.load_video("demo.mp4", target_fps=30)
latent_actions = extractor.extract_from_frames(frames)

# Create inference engine
inference = LatentGuidedInference(
    policy_path="checkpoint.pt",
    latent_actions=latent_actions,
    device="cuda"
)

# Use with your environment
import gymnasium as gym
env = gym.make("PushT-v0")
obs, _ = env.reset()

while True:
    action = inference.predict_action(obs)
    if action is None:
        break
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

## Understanding the Output

### Latent Actions File

The saved `latent_actions.pt` contains:
```python
{
    "latent_actions": Tensor,  # Shape: [T-1, lam_token_dim]
    "video_path": str,         # Original video path
    "num_frames": int,         # Number of frames processed
    "lam_model_path": str,     # LAM model used
}
```

Where:
- `T` = number of frames in video
- `T-1` = number of frame transitions (and latent actions)
- `lam_token_dim` = latent action dimension (e.g., 64 for 2 tokens × 32 dims)

### Latent Action Dimensions

For villa-x default config:
- `num_learned_tokens` = 2
- `action_latent_dim` = 32
- `lam_token_dim` = 2 × 32 = 64

Each latent action is a 64-dimensional vector encoding the "action" that caused the visual transition between consecutive frames.

## Common Options

### Video Processing

Resample video to 10 FPS:
```bash
--target-fps 10
```

Process only first 100 frames:
```bash
--max-frames 100
```

Use different LAM resolution:
```bash
--lam-resolution 224
```

### Inference Control

Limit inference steps:
```bash
--max-steps 500
```

Don't save video output:
```bash
--no-save-video
```

Use CPU instead of GPU:
```bash
--device cpu
```

## Troubleshooting

### "Could not open video file"
- Check video path is correct
- Install opencv-python: `pip install opencv-python`
- Try converting video format: `ffmpeg -i input.avi output.mp4`

### "LAM model not found"
- Ensure villa-x submodule is cloned: `git submodule update --init`
- Or the model will auto-download from HuggingFace on first use

### "Policy config mismatch"
- Ensure policy was trained with LAM conditioning
- Check policy config has `use_lam_conditioning=True`

### "CUDA out of memory"
- Reduce batch size in video processing
- Process fewer frames: `--max-frames 50`
- Use CPU: `--device cpu`

## Next Steps

1. **Experiment with different videos**: Try various demonstration styles
2. **Tune the FPS**: Match video FPS to your policy's training data
3. **Analyze latent actions**: Visualize what the LAM is encoding
4. **Real robot deployment**: Adapt robot interface for your hardware
5. **Multi-modal inputs**: Combine latent actions with other observations

## Example Use Cases

### 1. Task Variation
Use different demonstration videos to guide the same policy through task variations:
```bash
# Demo 1: Push object left
python ... --video-path demo_push_left.mp4

# Demo 2: Push object right  
python ... --video-path demo_push_right.mp4
```

### 2. Multi-Step Tasks
Break complex tasks into video segments with latent action guidance for each phase.

### 3. Human-Robot Collaboration
Use human demonstration videos to guide robot behavior in real-time.

### 4. Sim-to-Real Transfer
Extract latent actions from simulation videos, use them to guide real robot.

## Performance Tips

1. **Video Quality**: Higher quality videos → better latent actions
2. **FPS Matching**: Match video FPS to policy's expected observation rate
3. **GPU Usage**: Use GPU for LAM extraction (much faster)
4. **Caching**: Extract latent actions once, reuse multiple times

## Getting Help

- Check the full documentation: `OFFLINE_LATENT_IMITATION.md`
- Review example code in `lerobot/examples/offline_latent_imitation*.py`
- Open an issue on GitHub

Happy latent imitating! 🤖
