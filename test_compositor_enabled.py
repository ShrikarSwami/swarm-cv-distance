#!/usr/bin/env python3
"""Check if compositor is enabled in render settings"""

import bpy

scene = bpy.context.scene

print("=== Compositor Settings ===")
print(f"scene.use_nodes: {scene.use_nodes}")

# Check view layer compositor settings
vl = bpy.context.view_layer
print(f"\nView layer attributes with 'compositor' or 'pass':")
for attr in dir(vl):
    if 'compositor' in attr.lower() or 'pass' in attr.lower():
        try:
            val = getattr(vl, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

# Check if there's a compositor attribute on the scene
print(f"\nScene attributes with 'compositor':")
for attr in dir(scene):
    if 'compositor' in attr.lower():
        try:
            val = getattr(scene, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

# Check render engine
print(f"\nRender engine: {scene.render.engine}")

# Check if compositor is enabled in render
print(f"\nRender settings with 'compositor':")
for attr in dir(scene.render):
    if 'compositor' in attr.lower():
        try:
            val = getattr(scene.render, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass
