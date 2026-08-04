#!/usr/bin/env python3
"""Test EXR format options in Blender 5.2"""

import bpy

scene = bpy.context.scene

# Check available file formats
print("Available file formats:")
for fmt in ['OPEN_EXR', 'OPEN_EXR_MULTILAYER', 'PNG', 'TIFF']:
    try:
        scene.render.image_settings.file_format = fmt
        print(f"  {fmt}: AVAILABLE")
    except:
        print(f"  {fmt}: NOT AVAILABLE")

# Check EXR-specific settings
scene.render.image_settings.file_format = "OPEN_EXR"
print("\nEXR settings:")
for attr in dir(scene.render.image_settings):
    if 'exr' in attr.lower() or 'color' in attr.lower() or 'depth' in attr.lower():
        try:
            val = getattr(scene.render.image_settings, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass
