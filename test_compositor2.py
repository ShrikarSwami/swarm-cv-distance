#!/usr/bin/env python3
"""Test compositor node attributes in Blender 5.2"""

import bpy

scene = bpy.context.scene
scene.use_nodes = True

# Get or create compositor node group
tree = scene.compositing_node_group
if tree is None:
    tree = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
    scene.compositing_node_group = tree

# Create output file node
out = tree.nodes.new("CompositorNodeOutputFile")
print("CompositorNodeOutputFile attributes:")
for attr in dir(out):
    if not attr.startswith('_'):
        try:
            val = getattr(out, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass
