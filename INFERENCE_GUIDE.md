# Offline Latent Imitation Inference Guide

This guide explains how to run inference using your trained model with latent action conditioning from demonstration videos.

## Overview

The Offline Latent Imitation approach works as follows:

```
Demonstration Video → LAM Model → Latent Actions
                                       ↓
Current Observation + Latent Action → Policy → Robot Action
```

**Key Idea**: Instead of directly copying actions from a demonstration, we:
1. Extract **latent actions** (abstract action representations) from the demo video using LAM
2. Use these latent actions to **condition/guide** the policy during execution
3. The policy can adapt to the current observation while following the demonstration's intent

## Quick Start

### 1. Simple Example (Conceptual Demo)

Run the simple example to understand the workflow:

```bash
cd lerobot
python examples/offline_latent_imitation_inference_simple.py
```

This shows the core concepts without needing a real environment.

### 2. Full Inference Script

For actual robot control or simulation:

```bash
python scripts/offline_latent_imitation_inference.py \
    --policy-path Factory-Intelligence/Latent-Imitation-Model \
    --lam-model-path villa-x \
    --demo-video path/to/demonstration.mp4 \
    --env-name pusht \
    --device cuda \
    --num-episodes 10 \
    --render \
    --save-video \
    --output-dir outputs/inference
```

**Arguments:**
- `--policy-path`: Your trained policy (HuggingFace repo or local path)
- `--lam-model-path`: Path to LAM model (e.g., VILLA-X)
- `--demo-video`: Path to demonstration video file
- `--env-name`: Environment name (pusht, aloha, etc.)
- `--device`: cuda or cpu
- `--num-episodes`: How many rollouts to perform
- `--render`: Display environment during execution
- `--save-video`: Save rollout videos
- `--output-dir`: Where to save outputs

## How It Works

### Step 1: Extract Latent Actions from Demo

```python
# Load LAM model
lam_model = IgorModel.from_pretrained("villa-x")

# Load demo video frames (T, C, H, W)
demo_frames = load_video_frames("demo.mp4")

# Extract latent actions between consecutive frames
latent_actions = []
for t in range(len(demo_frames) - 1):
    # LAM's Inverse Dynamics Model: (s_t, s_{t+1}) -> latent_action
    latent_action = lam_model.idm(demo_frames[t], demo_frames[t+1])
    latent_actions.append(latent_action)
```

### Step 2: Use Latent Actions During Rollout

```python
# Load trained policy
policy = DiffusionPolicy.from_pretrained("your-model")

# Execute in environment
for step in range(num_steps):
    # Get current observation from environment
    obs = env.get_observation()
    
    # Get corresponding latent action from demo
    latent_action = latent_actions[step]
    
    # Combine observation + latent action
    batch = {
        'observation.state': obs,
        'observation.lam_tokens': latent_action
    }
    
    # Generate action conditioned on both
    action = policy.select_action(batch)
    
    # Execute action
    env.step(action)
```

## Preparation

### 1. Download LAM Model (VILLA-X)

The LAM model extracts latent actions from visual observations:

```bash
cd /path/to/pai-hack-26-pp
git submodule update --init villa-x

# Download VILLA-X pretrained weights
# (Add instructions based on where VILLA-X weights are hosted)
```

### 2. Prepare Demonstration Videos

Your demonstration video should:
- Show the task being performed correctly
- Have clear visibility of the robot/environment
- Match the camera view used during training (if applicable)
- Be in a standard video format (mp4, avi, etc.)

**Example**: For PushT, record a video of successfully pushing the T-block to the target.

### 3. Ensure Policy is Trained with LAM Conditioning

Your policy must be trained with `use_lam_conditioning=true`. Check the training config:

```bash
# Check if LAM was enabled during training
cat outputs/train/pusht_baseline/checkpoints/010000/pretrained_model/config.json | grep lam
```

## Advanced Usage

### Using Precomputed LAM Tokens

If you have a dataset with precomputed LAM tokens:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load dataset with precomputed tokens
dataset = LeRobotDataset("your-dataset")

# Get episode with precomputed tokens
episode = dataset.get_episode(0)
latent_actions = episode['observation.lam_tokens']  # Already computed!

# Use directly in rollout
# No need to run LAM model again
```

### Real Robot Deployment

For deploying on a real robot:

1. **Camera Setup**: Ensure camera matches training setup
2. **Action Scaling**: Verify action ranges match robot's capabilities
3. **Safety**: Implement safety checks and emergency stops
4. **Latency**: Consider compute time for LAM + Policy inference

```python
# Example with safety wrapper
import numpy as np

def safe_action(action, limits):
    """Clip actions to safe ranges"""
    return np.clip(action, limits['min'], limits['max'])

# In rollout loop
action = policy.select_action(batch)
action = safe_action(action, robot.action_limits)
robot.execute(action)
```

## Troubleshooting

### Issue: "LAM model not found"

**Solution**: Ensure VILLA-X is downloaded:
```bash
cd villa-x
# Download weights (follow VILLA-X instructions)
```

### Issue: "Dimension mismatch for LAM tokens"

**Solution**: Check LAM configuration matches training:
- `lam_num_learned_tokens`: Should match training config
- `lam_action_latent_dim`: Should match training config

### Issue: "Video has different resolution than expected"

**Solution**: LAM expects 224x224 images by default. The script automatically resizes, but you can adjust:
```bash
--lam-resolution 256  # If your LAM model uses different resolution
```

### Issue: "Policy performance is poor"

**Possible causes:**
1. **Demo quality**: Ensure demonstration is high-quality and representative
2. **Camera mismatch**: Demo camera should match policy's expected camera
3. **Distribution shift**: Environment conditions differ from training
4. **Latent action length**: Demo might be too short/long for the task

## Examples

### Example 1: PushT Environment

```bash
# Record a demonstration of PushT task
# (Use environment's recording feature or screen capture)

# Run inference
python scripts/offline_latent_imitation_inference.py \
    --policy-path Factory-Intelligence/Latent-Imitation-Model \
    --lam-model-path villa-x \
    --demo-video demos/pusht_expert.mp4 \
    --env-name pusht \
    --num-episodes 10 \
    --save-video
```

### Example 2: Custom Environment

```bash
# For your custom environment
python scripts/offline_latent_imitation_inference.py \
    --policy-path path/to/your/checkpoint \
    --lam-model-path villa-x \
    --demo-video path/to/demo.mp4 \
    --env-name your-env-name \
    --device cuda \
    --render
```

## Performance Metrics

The script reports:
- **Average Reward**: Mean reward across episodes
- **Average Steps**: Mean episode length
- **Success Rate**: Percentage of successful episodes (if environment provides success signal)

Example output:
```
==================================================
SUMMARY
==================================================
Average Reward: 0.845
Average Steps: 98.2
Success Rate: 80.0%
```

## Tips for Best Results

1. **High-Quality Demonstrations**: Use expert demonstrations that clearly show the task
2. **Consistent Camera**: Use the same camera angle as during training
3. **Multiple Demos**: Try different demonstration videos to see which works best
4. **Demo Length**: Match demo length to task horizon
5. **Environment Reset**: Ensure environment starts in similar state as demo

## Next Steps

- **Fine-tuning**: If performance isn't great, consider fine-tuning with more data
- **Demo Augmentation**: Try multiple different demonstrations
- **Hybrid Approaches**: Combine latent actions with other conditioning (language, goals)
- **Online Learning**: Use successful rollouts to continue training

## References

- [VILLA-X Paper](https://arxiv.org/abs/2312.xxxxx)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
