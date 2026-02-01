# Precomputed LAM Tokens Implementation

This document summarizes the implementation of precomputed LAM (Latent Action Model) tokens for the LeRobot Diffusion Policy.

## Overview

This feature allows precomputing LAM tokens from the VILLA-X model to speed up Diffusion Policy training. Instead of computing expensive LAM tokens on-the-fly during each training iteration, tokens can be computed once and reused.

## Changes Made

### 1. Configuration Updates (`lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py`)

Added two new configuration options:

```python
use_precomputed_lam_tokens: bool = False  # Use precomputed LAM tokens from dataset
lam_tokens_key: str = "observation.lam_tokens"  # Dataset key for precomputed LAM tokens
```

Updated validation logic to:
- Skip LAM model path requirement when using precomputed tokens
- Skip n_obs_steps validation for precomputed tokens
- Adjust `lam_conditioning_dim` calculation based on precomputed vs on-the-fly mode

### 2. Modeling Updates (`lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py`)

#### Added Constants
```python
OBS_LAM_TOKENS = "observation.lam_tokens"  # Key for precomputed LAM tokens
```

#### Updated Methods

**`_init_lam_encoder()`**: 
- Skip loading LAM model when using precomputed tokens
- Still initialize projection layer for token processing

**`_prepare_global_conditioning()`**: 
- Added logic to load precomputed tokens from batch
- Apply projection and aggregation to precomputed tokens
- Fall back to on-the-fly computation if not using precomputed mode

**`reset()`**: 
- Add queue for precomputed LAM tokens when enabled
- Otherwise use LAM images queue for on-the-fly computation

**`predict_action_chunk()`**: 
- Handle both precomputed tokens and LAM images in queue

**`select_action()`**: 
- Extract precomputed LAM tokens from batch if available
- Otherwise extract LAM images for on-the-fly computation

**`forward()`**: 
- Extract precomputed LAM tokens from batch during training
- Otherwise extract LAM images for on-the-fly computation

**`generate_actions()` and `compute_loss()`**: 
- Updated validation to check for correct LAM inputs based on mode

### 3. Precomputation Script (`lerobot/scripts/precompute_lam_tokens.py`)

Created a comprehensive script to precompute LAM tokens:

**Features:**
- Loads LeRobot datasets (v3 format)
- Loads VILLA-X LAM model
- Processes frames to extract latent action tokens
- Handles consecutive frame pairs for LAM
- Supports VQ tokens (discrete) or continuous tokens
- Batch processing for efficiency
- Saves tokens with metadata for reproducibility

**Usage:**
```bash
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --lam-model-path microsoft/villa-x \
    --camera-key observation.images.top \
    --output-dir ./data/pusht_lam
```

**Output Format:**
```python
{
    "tokens": torch.Tensor,      # [num_frames, lam_token_dim]
    "camera_key": str,            # Camera used
    "lam_model_path": str,        # LAM model used
    "lam_token_dim": int,         # Token dimension
    "use_vq_tokens": bool,        # VQ vs continuous
    "dataset_repo_id": str,       # Source dataset
    "num_frames": int,            # Total frames
}
```

### 4. Training Example (`lerobot/examples/train_with_precomputed_lam.py`)

Created a complete example showing:
- How to configure policy for precomputed tokens
- How to load dataset with precomputed tokens
- Complete training loop
- Checkpoint saving

### 5. Documentation (`lerobot/docs/PRECOMPUTED_LAM_TOKENS.md`)

Comprehensive documentation including:
- Quick start guide
- Detailed usage instructions
- Configuration options
- Implementation details
- Troubleshooting guide
- Performance comparisons
- Examples

## Architecture

### Token Extraction Flow

```
Dataset Frames → LAM Model → Latent Action Tokens
     ↓
Precompute Once
     ↓
Save to Disk (.pt file)
     ↓
Load During Training (fast!)
     ↓
Diffusion Policy Conditioning
```

### Training Flow Comparison

**On-the-fly (Original):**
```
Batch → Load Images → LAM Forward Pass → Extract Tokens → Project → Policy Forward
        (every iteration, expensive)
```

**Precomputed (New):**
```
Batch → Load Tokens → Project → Policy Forward
        (every iteration, fast!)
```

## Token Format

Each observation in the dataset gets a LAM token:

- **First frame**: Zero token (no previous frame to compare)
- **Frame i (i > 0)**: LAM token from (frame[i-1], frame[i])

This represents the latent action that transitions from frame[i-1] to frame[i].

**Dimensions:**
- Single token: `[lam_num_learned_tokens * lam_action_latent_dim]`
  - Default: `[2 * 32] = [64]`
- Per frame: `[1, 64]`
- Full dataset: `[total_frames, 64]`

**Aggregation modes:**
- `"latest"`: Use only most recent token → `[batch, 64]`
- `"concat"`: Concatenate all n_obs_steps tokens → `[batch, n_obs_steps * 64]`

## Configuration Matrix

| Mode | Config | Requirements |
|------|--------|--------------|
| **On-the-fly** | `use_lam_conditioning=True`<br>`use_precomputed_lam_tokens=False` | - LAM model path<br>- Overhead camera in dataset<br>- n_obs_steps ≥ 2 |
| **Precomputed** | `use_lam_conditioning=True`<br>`use_precomputed_lam_tokens=True` | - Precomputed tokens in dataset<br>- Matching token configuration |
| **Disabled** | `use_lam_conditioning=False` | None |

## File Structure

```
lerobot/
├── src/lerobot/policies/diffusion/
│   ├── configuration_diffusion.py  # Updated config
│   └── modeling_diffusion.py       # Updated model
├── scripts/
│   └── precompute_lam_tokens.py    # NEW: Precomputation script
├── examples/
│   └── train_with_precomputed_lam.py  # NEW: Training example
└── docs/
    └── PRECOMPUTED_LAM_TOKENS.md      # NEW: Documentation
```

## Usage Examples

### Example 1: Precompute for PushT

```bash
# Step 1: Precompute tokens
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top \
    --output-dir ./data/pusht_lam

# Step 2: Train with precomputed tokens
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id lerobot/pusht \
    --lam-tokens-path ./data/pusht_lam/lam_tokens.pt \
    --output-dir ./outputs/pusht_lam
```

### Example 2: Python API

```python
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Create config with precomputed tokens
config = DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
    lam_tokens_key="observation.lam_tokens",
    lam_num_learned_tokens=2,
    lam_action_latent_dim=32,
    lam_temporal_aggregation="latest",
)

# Create policy (no LAM model loaded!)
policy = DiffusionPolicy(config)

# Train as usual
# The policy will load precomputed tokens from the dataset
```

## Performance Benefits

**Estimated speedup** (compared to on-the-fly LAM computation):

- **Memory savings**: ~500-1000MB (LAM model not loaded)
- **Training speed**: ~15-30% faster per iteration
- **Setup cost**: One-time precomputation (minutes to hours depending on dataset size)

**Trade-offs:**
- ✅ Faster training iterations
- ✅ Lower memory usage
- ✅ Reproducible tokens
- ❌ Additional disk space for tokens
- ❌ One-time precomputation cost
- ❌ Less flexible (can't change LAM config per run)

## Future Enhancements

1. **Dataset Integration**: Add tokens as native LeRobot dataset v3 column
2. **Compression**: Compress tokens to reduce disk usage
3. **Batch Processing**: Parallel precomputation across multiple GPUs
4. **Streaming**: Support for streaming datasets
5. **Caching**: Intelligent caching for commonly used datasets

## Testing

To verify the implementation works:

```bash
# 1. Check configuration
python -c "from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig; \
  c = DiffusionConfig(use_lam_conditioning=True, use_precomputed_lam_tokens=True, \
  lam_num_learned_tokens=2, lam_action_latent_dim=32); print('Config OK')"

# 2. Check model initialization
python -c "from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy; \
  from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig; \
  c = DiffusionConfig(use_lam_conditioning=True, use_precomputed_lam_tokens=True, \
  lam_num_learned_tokens=2, lam_action_latent_dim=32); \
  p = DiffusionPolicy(c); print('Policy OK')"

# 3. Test precomputation script help
python lerobot/scripts/precompute_lam_tokens.py --help
```

## References

- **VILLA-X**: https://github.com/microsoft/villa-x
- **VILLA-X Paper**: https://arxiv.org/abs/2407.17453
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **LeRobot**: https://github.com/huggingface/lerobot

## Contributors

Implementation based on:
- LeRobot Diffusion Policy by HuggingFace
- VILLA-X LAM by Microsoft Research
- Diffusion Policy by Columbia AI & Robotics Lab
