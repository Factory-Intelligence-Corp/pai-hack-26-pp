# Offline Latent Imitation - Summary

## What Was Created

Three new inference scripts for offline latent imitation, along with comprehensive documentation:

### Scripts

1. **`offline_latent_imitation.py`** - Basic template
   - Video loading and latent action extraction
   - Template for policy integration
   - Good starting point for customization

2. **`offline_latent_imitation_robot.py`** - Full implementation
   - Complete simulation support (Gymnasium)
   - Real robot integration (template)
   - Modular classes for reuse
   - Multiple execution modes

3. **`offline_latent_imitation_minimal.py`** - Simple example
   - Self-contained demonstration
   - Shows core concept clearly
   - Educational/debugging tool

### Documentation

1. **`OFFLINE_LATENT_IMITATION.md`** - Complete documentation
   - Architecture and design
   - Integration with LeRobot
   - Troubleshooting guide
   - Advanced usage examples

2. **`QUICK_START_OFFLINE_LATENT_IMITATION.md`** - Quick start guide
   - 5-minute setup
   - Minimal examples
   - Common use cases
   - Command reference

3. **`OFFLINE_LATENT_IMITATION_SUMMARY.md`** - This file
   - Overview of all components
   - File structure
   - Quick reference

## Concept Overview

**Offline Latent Imitation** is a technique where:
1. A demonstration video shows desired behavior
2. LAM (Latent Action Model) extracts latent actions from the video
3. These latent actions condition a policy during inference
4. Each latent action guides one step of robot execution

**Why is this useful?**
- **Hierarchical control**: Video provides high-level guidance
- **Task variation**: Different videos → different behaviors
- **Demonstration-based**: Easy to specify desired behavior
- **Temporal structure**: Video provides action timing

## File Structure

```
lerobot/
├── examples/
│   ├── offline_latent_imitation.py              # Basic template
│   ├── offline_latent_imitation_robot.py        # Full implementation
│   ├── offline_latent_imitation_minimal.py      # Simple example
│   └── train_with_precomputed_lam.py            # Training (existing)
├── scripts/
│   └── precompute_lam_tokens.py                 # Precompute tokens (existing)

docs/
├── OFFLINE_LATENT_IMITATION.md                  # Full documentation
├── QUICK_START_OFFLINE_LATENT_IMITATION.md      # Quick start
└── OFFLINE_LATENT_IMITATION_SUMMARY.md          # This file
```

## Quick Reference

### Extract Latent Actions Only

```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo.mp4 \
    --policy-path checkpoint.pt \
    --mode extract_only
```

### Run in Simulation

```bash
python lerobot/examples/offline_latent_imitation_robot.py \
    --video-path demo.mp4 \
    --policy-path checkpoint.pt \
    --mode simulation \
    --env-name PushT-v0
```

### Minimal Example

```bash
python lerobot/examples/offline_latent_imitation_minimal.py \
    --video-path demo.mp4
```

## Key Classes

### `LatentActionExtractor`
Handles video → latent actions conversion:
- `load_video()` - Load and preprocess video
- `extract_from_frames()` - Extract latent actions using LAM

### `LatentGuidedInference`
Manages policy inference with latent conditioning:
- `get_next_latent_action()` - Get next latent in sequence
- `prepare_batch()` - Add latent to observation batch
- `predict_action()` - Get action from policy

## Workflow Integration

### Complete Pipeline

```
1. Dataset → Precompute LAM Tokens
   ├─ precompute_lam_tokens.py
   └─ Output: lam_tokens.pt

2. Train Policy with LAM Conditioning
   ├─ train_with_precomputed_lam.py
   └─ Output: policy checkpoint

3. Record Demonstration Video
   └─ Output: demo.mp4

4. Extract Latent Actions
   ├─ offline_latent_imitation_robot.py --mode extract_only
   └─ Output: latent_actions.pt

5. Run Inference
   ├─ offline_latent_imitation_robot.py --mode simulation/robot
   └─ Output: robot behavior guided by video
```

## Python API Example

```python
from lerobot.examples.offline_latent_imitation_robot import (
    LatentActionExtractor,
    LatentGuidedInference
)

# Extract latent actions
extractor = LatentActionExtractor("microsoft/villa-x")
frames = extractor.load_video("demo.mp4")
latent_actions = extractor.extract_from_frames(frames)

# Setup inference
inference = LatentGuidedInference(
    policy_path="checkpoint.pt",
    latent_actions=latent_actions
)

# Use in your loop
while True:
    obs = get_observation()
    action = inference.predict_action(obs)
    if action is None:
        break
    execute_action(action)
```

## How LAM Works

The LAM (Latent Action Model) from villa-x learns to:
1. **Encode visual transitions**: Given frames at time t and t+1
2. **Extract latent action**: What "caused" this transition
3. **Compress to latent space**: 64-dim vector (default config)

This latent action captures:
- High-level intent
- Motion characteristics
- Task-specific patterns

## Integration with LeRobot

These scripts work with LeRobot's:
- **DiffusionPolicy** with LAM conditioning
- **Dataset infrastructure** for training
- **Robot interfaces** for deployment
- **Evaluation pipelines** for testing

### Required Policy Config

```python
config = DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
    lam_tokens_key="observation.lam_tokens",
    lam_num_learned_tokens=2,
    lam_action_latent_dim=32,
)
```

## Common Use Cases

### 1. Task Variation
Use different demo videos to specify different task variants:
```bash
# Push left
python ... --video-path demo_push_left.mp4

# Push right
python ... --video-path demo_push_right.mp4
```

### 2. Sim-to-Real
Extract latents from simulation demos, use on real robot:
```bash
# Extract from sim
python ... --video-path sim_demo.mp4 --mode extract_only

# Use on robot
python ... --mode robot --robot-config real_robot.yaml
```

### 3. Human Demonstrations
Record human performing task, extract latents, guide robot:
```bash
# Extract from human demo
python ... --video-path human_demo.mp4 --mode extract_only

# Guide robot
python ... --mode robot
```

### 4. Multi-Phase Tasks
Break complex task into phases, use different videos per phase.

## Extending the Code

### Add Custom Video Processing

```python
class MyExtractor(LatentActionExtractor):
    def load_video(self, path):
        # Custom video loading
        frames = my_custom_loader(path)
        return frames
```

### Add Custom Robot Interface

```python
def run_my_robot_inference(inference):
    robot = MyRobot()
    robot.connect()
    
    while True:
        obs = robot.get_observation()
        action = inference.predict_action(obs)
        if action is None:
            break
        robot.execute(action)
```

### Add Custom Policy Support

```python
class MyGuidedInference(LatentGuidedInference):
    def _load_policy(self, path):
        # Custom policy loading
        return MyPolicy.from_pretrained(path)
```

## Troubleshooting Quick Ref

| Problem | Solution |
|---------|----------|
| Video won't load | Install opencv-python |
| LAM model not found | Init submodule or auto-downloads |
| Policy load fails | Adapt `_load_policy` method |
| CUDA OOM | Use `--device cpu` or `--max-frames` |
| Wrong latent shape | Check policy config matches LAM |

## Performance Tips

1. **Extract once, use many times**: Cache latent actions
2. **Use GPU**: 10-100x faster for LAM extraction
3. **Match FPS**: Align video FPS with policy expectations
4. **Quality matters**: Better video → better latent actions

## Next Steps

1. **Try the minimal example**: Get familiar with the concept
2. **Extract from your videos**: Test with your demonstrations
3. **Integrate with your policy**: Adapt loading code if needed
4. **Deploy on robot**: Implement robot interface
5. **Experiment**: Try different videos, settings, use cases

## References

- **LeRobot**: https://github.com/huggingface/lerobot
- **villa-x (LAM)**: https://github.com/microsoft/villa-x
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/

## Getting Help

1. Check the documentation:
   - `OFFLINE_LATENT_IMITATION.md` - Full details
   - `QUICK_START_OFFLINE_LATENT_IMITATION.md` - Quick start

2. Run the minimal example:
   - `offline_latent_imitation_minimal.py` - Self-contained demo

3. Review the code:
   - All scripts are well-commented
   - Classes are modular and reusable

4. Open an issue:
   - Include error messages
   - Describe your setup
   - Share relevant config

## Contributing

We welcome contributions:
- Bug fixes
- New features (e.g., real-time extraction)
- Robot interfaces
- Documentation improvements
- Example use cases

## License

Same as LeRobot (Apache 2.0)

---

**Summary**: You now have a complete offline latent imitation system integrated with LeRobot, including extraction scripts, inference engines, and comprehensive documentation. Start with the minimal example, then move to the full implementation for production use.
