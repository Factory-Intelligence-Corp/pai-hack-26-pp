# Offline Latent Imitation

This document describes the offline latent imitation inference scripts that allow you to use demonstration videos to guide robot policy execution through latent action conditioning.

## Overview

Offline latent imitation is a technique where:
1. A demonstration video is provided as input
2. Latent actions are extracted from consecutive video frames using the LAM (Latent Action Model)
3. The robot policy is conditioned on these latent actions during inference
4. Each latent action is sent one at a time to guide the robot's behavior

This approach enables:
- **Hierarchical control**: High-level guidance from video, low-level actions from policy
- **Demonstration-guided execution**: Use videos to structure policy behavior
- **Temporal action abstractions**: Videos provide temporal structure for the policy

## Scripts

### 1. `offline_latent_imitation.py`

Basic script that extracts latent actions from a video and provides a template for inference.

**Features:**
- Video loading with FPS resampling
- Latent action extraction using villa-x LAM model
- Template for policy loading and inference
- Outputs saved latent actions for reuse

**Usage:**
```bash
python lerobot/examples/offline_latent_imitation.py \
    --video-path ./demo_video.mp4 \
    --policy-path ./outputs/checkpoint.pt \
    --lam-model-path microsoft/villa-x \
    --output-dir ./inference_outputs
```

### 2. `offline_latent_imitation_robot.py`

Extended script with full robot/simulation integration.

**Features:**
- Complete simulation environment support (Gymnasium)
- Real robot integration (template provided)
- Video rendering and saving
- Modular design with reusable classes

**Usage Examples:**

Extract latent actions only:
```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo.mp4 \
    --policy-path checkpoint.pt \
    --mode extract_only
```

Run in simulation:
```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo_pusht.mp4 \
    --policy-path outputs/pusht_lam/checkpoint.pt \
    --mode simulation \
    --env-name PushT-v0 \
    --output-dir ./sim_outputs
```

Run with real robot (requires robot config):
```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo_real.mp4 \
    --policy-path outputs/robot_lam/checkpoint.pt \
    --mode robot \
    --robot-config configs/my_robot.yaml \
    --control-time-s 30.0
```

## How It Works

### Step 1: Video Processing

The video is loaded and optionally resampled to match the desired FPS:

```python
frames = load_video_frames(
    video_path="demo.mp4",
    target_fps=30,  # Optional resampling
    max_frames=100,  # Optional truncation
)
```

### Step 2: Latent Action Extraction

Consecutive frame pairs are processed through the LAM model's inverse dynamics model (IDM):

```python
# For frames at time t and t+1
frame_pair = [frame_t, frame_t+1]

# Extract latent action that "caused" the transition
latent_action = lam_model.idm(frame_pair)
```

The LAM model learns to encode the action that would cause the visual transition from frame_t to frame_t+1 into a compact latent representation.

### Step 3: Policy Conditioning

During inference, each latent action conditions the policy:

```python
for latent_action in latent_actions:
    # Get current observation from environment
    obs = env.get_observation()
    
    # Prepare batch with LAM conditioning
    batch = {
        **obs,
        "observation.lam_tokens": latent_action
    }
    
    # Policy predicts action conditioned on latent
    action = policy.select_action(batch)
    
    # Execute action
    env.step(action)
```

## Architecture

### Classes

#### `LatentActionExtractor`
Handles video loading and latent action extraction.

**Methods:**
- `load_video(video_path, target_fps, max_frames)` - Load video frames
- `extract_from_frames(frames)` - Extract latent actions from frames

#### `LatentGuidedInference`
Manages policy loading and latent-guided inference.

**Methods:**
- `get_next_latent_action()` - Get next latent action in sequence
- `prepare_batch(observation, latent_action)` - Prepare batch with LAM conditioning
- `predict_action(observation)` - Predict action conditioned on next latent action
- `reset()` - Reset to beginning of sequence

## Integration with LeRobot

These scripts integrate with LeRobot's existing infrastructure:

### Policy Compatibility

The scripts work with any LeRobot policy that supports LAM conditioning:
- DiffusionPolicy with `use_lam_conditioning=True`
- Custom policies implementing LAM conditioning

### Required Policy Configuration

Your policy must be configured with:
```python
config = DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
    lam_tokens_key="observation.lam_tokens",
    lam_num_learned_tokens=2,
    lam_action_latent_dim=32,
    lam_temporal_aggregation="latest",
)
```

### Dataset Integration

While these scripts work with videos, they're compatible with the dataset-based training:
1. Train with precomputed LAM tokens (see `train_with_precomputed_lam.py`)
2. Use trained policy for video-guided inference

## Workflow Example

Complete workflow from training to inference:

### 1. Precompute LAM tokens for training dataset
```bash
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --lam-model-path microsoft/villa-x \
    --camera-key observation.images.top
```

### 2. Train policy with LAM conditioning
```bash
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id lerobot/pusht \
    --lam-tokens-path ./data/pusht_with_lam/lam_tokens.pt \
    --output-dir ./outputs/pusht_lam
```

### 3. Record demonstration video
Use any method to record a demonstration video showing the desired behavior.

### 4. Run offline latent imitation
```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demonstration.mp4 \
    --policy-path ./outputs/pusht_lam/checkpoint_epoch_100.pt \
    --mode simulation \
    --env-name PushT-v0
```

## Advanced Usage

### Custom Video Processing

You can customize video processing:
```python
extractor = LatentActionExtractor(
    lam_model_path="microsoft/villa-x",
    lam_resolution=224,
    device="cuda"
)

# Custom video loading
frames = custom_load_function()

# Extract latent actions
latent_actions = extractor.extract_from_frames(frames)
```

### Reusing Precomputed Latent Actions

Extract once, use multiple times:
```bash
# Extract and save
python ... --mode extract_only --output-dir ./latents/

# Later, load from saved file
latent_actions = torch.load("./latents/latent_actions.pt")["latent_actions"]
```

### Robot-Specific Integration

For real robot deployment, implement the robot interface:
```python
def run_robot_inference(robot_config, inference, output_dir):
    from lerobot.robots import make_robot_from_config
    
    robot = make_robot_from_config(robot_config)
    robot.connect()
    
    try:
        while True:
            obs = robot.get_observation()
            action = inference.predict_action(obs)
            if action is None:
                break
            robot.send_action(action)
    finally:
        robot.disconnect()
```

## Command Line Arguments

### Common Arguments

- `--video-path` (required): Path to demonstration video
- `--policy-path` (required): Path to trained policy checkpoint
- `--lam-model-path`: LAM model path (default: "microsoft/villa-x")
- `--output-dir`: Output directory (default: "./inference_outputs")
- `--device`: Device to use (default: "cuda" if available)

### Video Processing Arguments

- `--target-fps`: Resample video to target FPS
- `--max-frames`: Maximum frames to process from video
- `--lam-resolution`: LAM model input resolution (default: 224)

### Execution Mode Arguments (`offline_latent_imitation_robot.py`)

- `--mode`: Execution mode: `extract_only`, `simulation`, or `robot`
- `--env-name`: Gym environment name (simulation mode)
- `--robot-config`: Robot config path (robot mode)
- `--max-steps`: Maximum inference steps
- `--control-time-s`: Control time in seconds (robot mode)
- `--fps`: Control loop FPS

### Output Arguments

- `--no-render`: Disable rendering during inference
- `--no-save-video`: Don't save inference video

## Troubleshooting

### Video Loading Issues

**Problem**: Video fails to load
**Solution**: Ensure OpenCV is installed with video support:
```bash
pip install opencv-python
```

### LAM Model Issues

**Problem**: LAM model not found
**Solution**: Ensure villa-x submodule is initialized:
```bash
git submodule update --init --recursive
```

### Policy Loading Issues

**Problem**: Policy fails to load from checkpoint
**Solution**: The checkpoint format varies. You may need to adapt the `_load_policy` method:
```python
def _load_policy(self, policy_path):
    checkpoint = torch.load(policy_path)
    # Adapt based on your checkpoint structure
    config = checkpoint["config"]  # or however config is stored
    policy = DiffusionPolicy(config)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy
```

### LAM Conditioning Issues

**Problem**: Policy doesn't seem to use latent actions
**Solution**: Verify policy config has LAM conditioning enabled:
```python
assert policy.config.use_lam_conditioning == True
assert policy.config.use_precomputed_lam_tokens == True
```

## Future Extensions

Possible extensions to this approach:

1. **Interactive latent editing**: Allow modifying latent actions before inference
2. **Multi-video blending**: Combine latent actions from multiple demonstrations
3. **Latent action interpolation**: Smooth transitions between latent actions
4. **Real-time latent extraction**: Extract latent actions from live camera feed
5. **Latent action visualization**: Visualize what each latent action represents

## References

- LeRobot: https://github.com/huggingface/lerobot
- villa-x (LAM): https://github.com/microsoft/villa-x
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

## Contributing

To extend these scripts:
1. Add custom video processing in `LatentActionExtractor`
2. Add custom robot interfaces in `run_robot_inference`
3. Add custom policy types in `LatentGuidedInference._load_policy`
4. Submit pull requests with improvements!
