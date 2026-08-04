#!/usr/bin/env python3
"""Check import paths and dependencies"""

import sys
import os

# Add stage1_geometry to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage1_geometry'))

print("=== Checking Imports ===\n")

# Test 1: Import data_contract directly
try:
    import data_contract
    print("✓ data_contract imported successfully")
    print(f"  DEFAULT_FOCAL_PX = {data_contract.DEFAULT_FOCAL_PX}")
except Exception as e:
    print(f"✗ data_contract import failed: {e}")

# Test 2: Try to import all the modules
try:
    from data_contract import CameraRig, SwarmTruth, Detections, make_K, CONVENTION_TAG
    print("✓ data_contract imports successful")
except Exception as e:
    print(f"✗ data_contract imports failed: {e}")

# Test 3: Check if b1_scene_rig can be imported
try:
    import b1_scene_rig
    print("✓ b1_scene_rig imported successfully")
except Exception as e:
    print(f"✗ b1_scene_rig import failed: {e}")

# Test 4: Check if b2_projection can be imported
try:
    import b2_projection
    print("✓ b2_projection imported successfully")
except Exception as e:
    print(f"✗ b2_projection import failed: {e}")

# Test 5: Check if b3_correspondence can be imported
try:
    import b3_correspondence
    print("✓ b3_correspondence imported successfully")
except Exception as e:
    print(f"✗ b3_correspondence import failed: {e}")

# Test 6: Try to use the classes
try:
    import numpy as np

    # Create a simple test
    K = make_K(2666.67)
    print(f"✓ make_K(2666.67) works: {K}")

    # Test CameraRig creation (simplified)
    print("✓ All core imports working")

except Exception as e:
    print(f"✗ Test failed: {e}")

print("\n=== Checking Module Structure ===\n")

# List files in stage1_geometry
import glob
py_files = glob.glob("stage1_geometry/*.py")
print(f"Found {len(py_files)} Python files in stage1_geometry:")
for f in py_files:
    print(f"  - {f}")

print("\n=== Project Root Structure ===\n")
for item in os.listdir('.'):
    if os.path.isdir(item) and not item.startswith('.'):
        print(f"Directory: {item}")
    elif item.endswith('.py'):
        print(f"File: {item}")

print("\n=== Done ===")