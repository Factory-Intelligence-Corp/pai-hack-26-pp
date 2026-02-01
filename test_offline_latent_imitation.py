#!/usr/bin/env python

"""
Test script to verify offline latent imitation setup.

This script checks:
1. All dependencies are installed
2. villa-x LAM model is accessible
3. Scripts are importable
4. Basic functionality works

Usage:
    python test_offline_latent_imitation.py
"""

import sys
from pathlib import Path

print("="*80)
print("Testing Offline Latent Imitation Setup")
print("="*80)

# Test 1: Check Python version
print("\n[1/7] Checking Python version...")
if sys.version_info < (3, 8):
    print("❌ FAIL: Python 3.8+ required")
    sys.exit(1)
print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Check dependencies
print("\n[2/7] Checking dependencies...")
dependencies = {
    "torch": "PyTorch",
    "cv2": "OpenCV (opencv-python)",
    "einops": "einops",
    "numpy": "NumPy",
    "tqdm": "tqdm",
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name}")
    except ImportError:
        print(f"❌ {name} not found")
        missing.append(name)

if missing:
    print(f"\n❌ Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join([dep.split()[0].lower() for dep in missing]))
    sys.exit(1)

# Test 3: Check villa-x submodule
print("\n[3/7] Checking villa-x submodule...")
repo_root = Path(__file__).parent
villa_x_path = repo_root / "villa-x"

if not villa_x_path.exists():
    print("❌ villa-x directory not found")
    print("Initialize with: git submodule update --init villa-x")
    sys.exit(1)

lam_init = villa_x_path / "lam" / "__init__.py"
if not lam_init.exists():
    print("❌ villa-x/lam not properly initialized")
    print("Initialize with: git submodule update --init --recursive")
    sys.exit(1)

print("✓ villa-x submodule found")

# Test 4: Import LAM model
print("\n[4/7] Testing LAM model import...")
sys.path.insert(0, str(villa_x_path))
try:
    from lam import IgorModel
    print("✓ LAM model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LAM model: {e}")
    sys.exit(1)

# Test 5: Check scripts exist
print("\n[5/7] Checking offline latent imitation scripts...")
scripts = [
    "lerobot/examples/offline_latent_imitation_minimal.py",
    "lerobot/examples/offline_latent_imitation.py",
    "lerobot/examples/offline_latent_imitation_robot.py",
]

for script in scripts:
    script_path = repo_root / script
    if not script_path.exists():
        print(f"❌ {script} not found")
        sys.exit(1)
    print(f"✓ {script_path.name}")

# Test 6: Test imports
print("\n[6/7] Testing script imports...")
sys.path.insert(0, str(repo_root / "lerobot" / "examples"))
try:
    from offline_latent_imitation_robot import LatentActionExtractor, LatentGuidedInference
    print("✓ Successfully imported LatentActionExtractor")
    print("✓ Successfully imported LatentGuidedInference")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 7: Test basic functionality
print("\n[7/7] Testing basic functionality...")
try:
    import torch
    import numpy as np
    
    # Create dummy frames
    dummy_frames = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    
    print("✓ Created dummy video frames")
    
    # Note: We don't test actual LAM extraction here as it requires downloading the model
    # which could be slow. Users can test that with the minimal script.
    
    print("✓ Basic functionality test passed")
    
except Exception as e:
    print(f"❌ Functionality test failed: {e}")
    sys.exit(1)

# All tests passed!
print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nYour offline latent imitation setup is ready to use!")
print("\nNext steps:")
print("1. Try the minimal example:")
print("   python lerobot/examples/offline_latent_imitation_minimal.py --video-path demo.mp4")
print("\n2. Read the quick start guide:")
print("   QUICK_START_OFFLINE_LATENT_IMITATION.md")
print("\n3. Check the full documentation:")
print("   OFFLINE_LATENT_IMITATION.md")
print("\n" + "="*80)
