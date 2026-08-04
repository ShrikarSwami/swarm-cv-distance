#!/usr/bin/env python3
"""Test render passes in Blender 5.2"""

import bpy

scene = bpy.context.scene

# Check view layer
vl = bpy.context.view_layer
print("View layer attributes:")
for attr in dir(vl):
    if 'pass' in attr.lower() or 'object' in attr.lower():
        try:
            val = getattr(vl, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

# Check if there's a way to enable Object Index pass
print("\nView layer pass attributes:")
for attr in dir(vl):
    if 'pass' in attr.lower():
        try:
            val = getattr(vl, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass
