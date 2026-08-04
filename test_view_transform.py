#!/usr/bin/env python3
"""Test view transform settings in Blender 5.2"""

import bpy

scene = bpy.context.scene

# Check current view transform settings
print("=== View Transform Settings ===")
print(f"view_transform: {scene.view_settings.view_transform}")
print(f"look_transform: {scene.view_settings.look_transform}")
print(f"exposure: {scene.view_settings.exposure}")
print(f"gamma: {scene.view_settings.gamma}")
print(f"white_balance: {scene.view_settings.white_balance}")

# Check available view transforms
print("\nAvailable view transforms:")
for vt in ['Standard', 'AgX', 'Filmic', 'Raw', 'False Color']:
    try:
        scene.view_settings.view_transform = vt
        print(f"  {vt}: AVAILABLE")
    except:
        print(f"  {vt}: NOT AVAILABLE")

# Reset to original
scene.view_settings.view_transform = "Standard"
