#!/usr/bin/env python3
"""Test ALL file format enum values in Blender 5.2"""

import bpy

scene = bpy.context.scene

# List ALL available file formats
print("=== ALL Available File Formats ===")
all_formats = []
for attr in dir(scene.render.image_settings):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(scene.render.image_settings, attr)
        if not callable(val) and attr != 'bl_rna':
            # Try to set it as file_format
            try:
                scene.render.image_settings.file_format = attr
                all_formats.append(attr)
                print(f"  {attr}: AVAILABLE")
            except:
                pass
    except:
        pass

print(f"\nTotal formats: {len(all_formats)}")

# Specifically check EXR formats
print("\n=== EXR Formats ===")
for fmt in ['OPEN_EXR', 'OPEN_EXR_MULTILAYER', 'OPEN_EXR_MULTILAYER_EXR']:
    try:
        scene.render.image_settings.file_format = fmt
        print(f"  {fmt}: AVAILABLE ✓")
    except Exception as e:
        print(f"  {fmt}: NOT AVAILABLE ✗ ({e})")

# Check CompositorNodeOutputFile format options
print("\n=== CompositorNodeOutputFile Format Options ===")
scene.use_nodes = True
tree = scene.compositing_node_group
if tree is None:
    tree = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
    scene.compositing_node_group = tree

out = tree.nodes.new("CompositorNodeOutputFile")
print("CompositorNodeOutputFile format options:")
for fmt in ['OPEN_EXR', 'OPEN_EXR_MULTILAYER', 'PNG', 'TIFF']:
    try:
        out.format.file_format = fmt
        print(f"  {fmt}: AVAILABLE ✓")
    except Exception as e:
        print(f"  {fmt}: NOT AVAILABLE ✗ ({e})")
