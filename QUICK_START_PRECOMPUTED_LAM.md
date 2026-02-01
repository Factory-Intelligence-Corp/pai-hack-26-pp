# Quick Start: Precomputed LAM Tokens

## TL;DR

Speed up Diffusion Policy training by precomputing LAM tokens once instead of computing them every iteration.

---

## 🚀 3-Step Setup

### Step 1: Precompute Tokens (One-Time)

```bash
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top \
    --output-dir ./data/pusht_lam
```

### Step 2: Configure Policy

```python
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

config = DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,  # ← Key setting!
    lam_tokens_key="observation.lam_tokens",
)
```

### Step 3: Train (Faster!)

```bash
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id lerobot/pusht \
    --lam-tokens-path ./data/pusht_lam/lam_tokens.pt
```

---

## 💡 Why Use This?

| Benefit | Description |
|---------|-------------|
| ⚡ **15-30% Faster** | No LAM computation during training |
| 💾 **Less Memory** | LAM model not loaded (~500-1000 MB saved) |
| 🔄 **Reusable** | Compute once, train many times |
| 📊 **Reproducible** | Same tokens every run |

---

## 🎯 Common Use Cases

### Use Case 1: Standard Training
```bash
# Precompute
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top

# Train
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id lerobot/pusht \
    --lam-tokens-path ./data/pusht_lam/lam_tokens.pt
```

### Use Case 2: Hyperparameter Search
```bash
# Precompute once
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top

# Try different learning rates (all fast!)
for lr in 1e-4 5e-5 1e-5; do
    python lerobot/examples/train_with_precomputed_lam.py \
        --dataset-repo-id lerobot/pusht \
        --lam-tokens-path ./data/pusht_lam/lam_tokens.pt \
        --learning-rate $lr \
        --output-dir ./outputs/lr_$lr
done
```

### Use Case 3: Custom Dataset
```bash
# Find your camera key first
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('your/dataset')
frame = dataset[0]
print([k for k in frame.keys() if 'image' in k])
"

# Then precompute
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id your/dataset \
    --camera-key observation.images.YOUR_CAMERA
```

---

## 🔧 Configuration Cheatsheet

### Minimal Config (Precomputed)
```python
DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
)
```

### Full Config (Precomputed)
```python
DiffusionConfig(
    # LAM conditioning
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
    lam_tokens_key="observation.lam_tokens",
    
    # Token settings (must match precomputation!)
    lam_num_learned_tokens=2,
    lam_action_latent_dim=32,
    lam_temporal_aggregation="latest",  # or "concat"
)
```

### On-the-Fly Config (Original)
```python
DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=False,  # ← On-the-fly
    lam_model_path="microsoft/villa-x",
    lam_camera_key="observation.images.overhead",
)
```

---

## 📋 Command Reference

### Precomputation Script

```bash
# Basic usage
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id DATASET \
    --camera-key CAMERA

# With options
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top \
    --output-dir ./data/pusht_lam \
    --device cuda \
    --batch-size 32

# Custom LAM settings
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top \
    --lam-num-learned-tokens 4 \
    --lam-action-latent-dim 64 \
    --no-vq-tokens  # Use continuous instead of VQ

# Process specific episodes
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top \
    --episodes 0 1 2 3 4  # Only these episodes

# Help
python lerobot/scripts/precompute_lam_tokens.py --help
```

### Training Script

```bash
# Basic usage
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id DATASET \
    --lam-tokens-path TOKENS_PATH

# With options
python lerobot/examples/train_with_precomputed_lam.py \
    --dataset-repo-id lerobot/pusht \
    --lam-tokens-path ./data/pusht_lam/lam_tokens.pt \
    --output-dir ./outputs/pusht_lam \
    --num-epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --device cuda

# Help
python lerobot/examples/train_with_precomputed_lam.py --help
```

---

## ⚠️ Common Issues

### Issue 1: Camera key not found
```bash
# Check available cameras
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('lerobot/pusht')
print([k for k in dataset[0].keys() if 'image' in k])
"
```

### Issue 2: Token dimension mismatch
**Problem**: Config doesn't match precomputed tokens

**Solution**: Use same settings for precomputation and training:
- `lam_num_learned_tokens` (default: 2)
- `lam_action_latent_dim` (default: 32)
- `lam_temporal_aggregation` (default: "latest")

### Issue 3: Out of memory
```bash
# Reduce batch size during precomputation
python lerobot/scripts/precompute_lam_tokens.py \
    --dataset-repo-id lerobot/pusht \
    --camera-key observation.images.top \
    --batch-size 16  # ← Reduce this
```

---

## 📚 Documentation

- **User Guide**: `lerobot/docs/PRECOMPUTED_LAM_TOKENS.md`
- **Technical Docs**: `PRECOMPUTED_LAM_IMPLEMENTATION.md`
- **Summary**: `IMPLEMENTATION_SUMMARY.md`
- **This Guide**: `QUICK_START_PRECOMPUTED_LAM.md`

---

## 🧪 Verify Installation

```bash
# Test configuration
python -c "
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
c = DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
)
print(f'✓ Config OK: lam_token_dim={c.lam_token_dim}')
"

# Test policy
python -c "
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
c = DiffusionConfig(
    use_lam_conditioning=True,
    use_precomputed_lam_tokens=True,
)
p = DiffusionPolicy(c)
print(f'✓ Policy OK: LAM model loaded = {p.diffusion.lam_model is not None}')
"

# Run full test suite
python test_precomputed_lam.py
```

---

## 🎯 Decision Tree

```
Do you want to use LAM conditioning?
├─ No → Set use_lam_conditioning=False
└─ Yes → Do you want to precompute?
    ├─ Yes (faster training, one-time setup)
    │   ├─ Step 1: Run precompute_lam_tokens.py
    │   └─ Step 2: Set use_precomputed_lam_tokens=True
    └─ No (slower training, no setup)
        └─ Set use_precomputed_lam_tokens=False
           and provide lam_model_path
```

---

## 🔥 Pro Tips

1. **Precompute early**: Do it once for each dataset you'll use
2. **Reuse tokens**: Same tokens work for all hyperparameter experiments
3. **Check dimensions**: Token dim = `num_learned_tokens × action_latent_dim`
4. **Save tokens**: Keep them for future experiments
5. **GPU recommended**: Precomputation is faster on GPU

---

## ✅ Checklist

Before training with precomputed tokens:

- [ ] Precomputed LAM tokens saved to disk
- [ ] Verified camera key exists in dataset
- [ ] Config has `use_precomputed_lam_tokens=True`
- [ ] Token dimensions match between precomputation and config
- [ ] Dataset can load tokens (check path)

---

**Ready to train faster? Start with Step 1! 🚀**
