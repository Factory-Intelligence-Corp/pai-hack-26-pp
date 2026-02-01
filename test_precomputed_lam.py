#!/usr/bin/env python

"""
Quick test script to verify precomputed LAM tokens implementation.

This script tests:
1. Configuration creation with precomputed tokens
2. Policy initialization without LAM model
3. Token dimension calculations
"""

import torch


def test_config():
    """Test configuration with precomputed LAM tokens."""
    print("=" * 80)
    print("Test 1: Configuration")
    print("=" * 80)
    
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    
    # Test with precomputed tokens enabled
    config = DiffusionConfig(
        use_lam_conditioning=True,
        use_precomputed_lam_tokens=True,
        lam_tokens_key="observation.lam_tokens",
        lam_num_learned_tokens=2,
        lam_action_latent_dim=32,
        lam_temporal_aggregation="latest",
        n_obs_steps=2,
    )
    
    print(f"✓ Config created successfully")
    print(f"  - use_lam_conditioning: {config.use_lam_conditioning}")
    print(f"  - use_precomputed_lam_tokens: {config.use_precomputed_lam_tokens}")
    print(f"  - lam_token_dim: {config.lam_token_dim}")
    print(f"  - lam_conditioning_dim: {config.lam_conditioning_dim}")
    
    assert config.lam_token_dim == 64, f"Expected 64, got {config.lam_token_dim}"
    assert config.lam_conditioning_dim == 64, f"Expected 64, got {config.lam_conditioning_dim}"
    
    # Test with concat aggregation
    config_concat = DiffusionConfig(
        use_lam_conditioning=True,
        use_precomputed_lam_tokens=True,
        lam_tokens_key="observation.lam_tokens",
        lam_num_learned_tokens=2,
        lam_action_latent_dim=32,
        lam_temporal_aggregation="concat",
        n_obs_steps=3,
    )
    
    expected_dim = 64 * 3  # n_obs_steps * token_dim for precomputed
    assert config_concat.lam_conditioning_dim == expected_dim, \
        f"Expected {expected_dim}, got {config_concat.lam_conditioning_dim}"
    
    print(f"✓ Concat aggregation: {config_concat.lam_conditioning_dim}")
    print()
    
    return config


def test_policy_init(config):
    """Test policy initialization with precomputed tokens."""
    print("=" * 80)
    print("Test 2: Policy Initialization")
    print("=" * 80)
    
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    
    # Create policy (should not load LAM model)
    policy = DiffusionPolicy(config)
    
    print(f"✓ Policy created successfully")
    print(f"  - LAM model loaded: {policy.diffusion.lam_model is not None}")
    print(f"  - LAM projection layer exists: {hasattr(policy.diffusion, 'lam_projection')}")
    
    assert policy.diffusion.lam_model is None, "LAM model should not be loaded for precomputed tokens"
    assert hasattr(policy.diffusion, 'lam_projection'), "LAM projection layer should exist"
    
    print()
    return policy


def test_queues(policy):
    """Test queue initialization."""
    print("=" * 80)
    print("Test 3: Queue Initialization")
    print("=" * 80)
    
    from lerobot.policies.diffusion.modeling_diffusion import OBS_LAM_TOKENS, OBS_LAM_IMAGES
    
    # Reset should create queues
    policy.reset()
    
    print(f"✓ Queues initialized")
    print(f"  - Queues: {list(policy._queues.keys())}")
    
    # Should have LAM tokens queue, not LAM images
    assert OBS_LAM_TOKENS in policy._queues, f"{OBS_LAM_TOKENS} should be in queues"
    assert OBS_LAM_IMAGES not in policy._queues, f"{OBS_LAM_IMAGES} should not be in queues"
    
    print(f"  - {OBS_LAM_TOKENS} queue max length: {policy._queues[OBS_LAM_TOKENS].maxlen}")
    print()


def test_forward_pass(policy):
    """Test forward pass with mock precomputed tokens."""
    print("=" * 80)
    print("Test 4: Forward Pass with Mock Tokens")
    print("=" * 80)
    
    from lerobot.policies.diffusion.modeling_diffusion import OBS_LAM_TOKENS
    
    batch_size = 4
    n_obs_steps = policy.config.n_obs_steps
    horizon = policy.config.horizon
    
    # Create mock batch with precomputed tokens
    batch = {
        "observation.state": torch.randn(batch_size, n_obs_steps, 2),
        "observation.images": torch.randn(batch_size, n_obs_steps, 1, 3, 96, 96),
        OBS_LAM_TOKENS: torch.randn(batch_size, n_obs_steps, 64),  # Mock precomputed tokens
        "action": torch.randn(batch_size, horizon, 2),
        "action_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
    }
    
    try:
        # Forward pass
        loss, _ = policy.forward(batch)
        
        print(f"✓ Forward pass successful")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Loss shape: {loss.shape}")
        
        assert loss.numel() == 1, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise
    
    print()


def test_comparison():
    """Compare configurations with/without precomputed tokens."""
    print("=" * 80)
    print("Test 5: Configuration Comparison")
    print("=" * 80)
    
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    
    # On-the-fly config
    config_otf = DiffusionConfig(
        use_lam_conditioning=True,
        use_precomputed_lam_tokens=False,
        lam_model_path="microsoft/villa-x",  # Required for on-the-fly
        lam_camera_key="observation.images.overhead",
        lam_temporal_aggregation="concat",
        n_obs_steps=3,
    )
    
    # Precomputed config
    config_pre = DiffusionConfig(
        use_lam_conditioning=True,
        use_precomputed_lam_tokens=True,
        lam_tokens_key="observation.lam_tokens",
        lam_temporal_aggregation="concat",
        n_obs_steps=3,
    )
    
    print(f"On-the-fly LAM conditioning dim: {config_otf.lam_conditioning_dim}")
    print(f"  - Calculated as: (n_obs_steps - 1) * token_dim = (3-1) * 64 = {(3-1) * 64}")
    
    print(f"Precomputed LAM conditioning dim: {config_pre.lam_conditioning_dim}")
    print(f"  - Calculated as: n_obs_steps * token_dim = 3 * 64 = {3 * 64}")
    
    # On-the-fly has n_obs_steps - 1 tokens (consecutive pairs)
    # Precomputed has n_obs_steps tokens (one per observation)
    assert config_otf.lam_conditioning_dim == (3 - 1) * 64
    assert config_pre.lam_conditioning_dim == 3 * 64
    
    print(f"✓ Dimension calculations correct")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  Precomputed LAM Tokens Implementation Test Suite".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()
    
    try:
        # Run tests
        config = test_config()
        policy = test_policy_init(config)
        test_queues(policy)
        test_forward_pass(policy)
        test_comparison()
        
        # Summary
        print("=" * 80)
        print("✅ All Tests Passed!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ Configuration with precomputed tokens works")
        print("  ✓ Policy initializes without loading LAM model")
        print("  ✓ Queues are correctly set up")
        print("  ✓ Forward pass works with mock precomputed tokens")
        print("  ✓ Token dimension calculations are correct")
        print()
        print("Next steps:")
        print("  1. Precompute tokens for your dataset:")
        print("     python lerobot/scripts/precompute_lam_tokens.py --help")
        print("  2. Train with precomputed tokens:")
        print("     python lerobot/examples/train_with_precomputed_lam.py --help")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Test Failed!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
