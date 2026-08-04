#!/usr/bin/env python3
"""Test compositor node inputs/outputs in Blender 5.2"""

import bpy

scene = bpy.context.scene
scene.use_nodes = True

# Get or create compositor node group
tree = scene.compositing_node_group
if tree is None:
    tree = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
    scene.compositing_node_group = tree
tree.nodes.clear()

# Create render layers node
rl = tree.nodes.new("CompositorNodeRLayers")
print("CompositorNodeRLayers outputs:")
for output in rl.outputs:
    print(f"  {output.name}: {output.type}")

# Create output file node
out = tree.nodes.new("CompositorNodeOutputFile")
print("\nCompositorNodeOutputFile inputs:")
for inp in out.inputs:
    print(f"  {inp.name}: {inp.type}")
