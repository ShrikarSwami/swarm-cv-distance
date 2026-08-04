#!/usr/bin/env python3
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

print("Current working directory:", os.getcwd())
print("Python path includes:", sys.path[:3])

# Try to import data_contract
try:
    from stage1_geometry import data_contract
    print("✓ Successfully imported data_contract from stage1_geometry")
    print(f"  DEFAULT_FOCAL_PX = {data_contract.DEFAULT_FOCAL_PX}")
except Exception as e:
    print(f"✗ Failed to import data_contract from stage1_geometry: {e}")

# Try direct import
try:
    import data_contract
    print("✓ Successfully imported data_contract directly")
    print(f"  DEFAULT_FOCAL_PX = {data_contract.DEFAULT_FOCAL_PX}")
except Exception as e:
    print(f"✗ Failed to import data_contract directly: {e}")

# Try relative import
try:
    from .stage1_geometry import data_contract
    print("✓ Successfully imported with relative import")
except:
    print("Note: Relative import requires being in a package context")

# Check if we can access the module by filename
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_contract", "stage1_geometry/data_contract.py")
    if spec:
        print("✓ data_contract.py file found and spec can be created")
    else:
        print("✗ data_contract.py file not found or spec creation failed")
except Exception as e:
    print(f"✗ Error checking file: {e}")