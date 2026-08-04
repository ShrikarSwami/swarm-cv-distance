#!/usr/bin/env python3
"""Test compositor access in Blender 5.2"""

import bpy

scene = bpy.context.scene
print('Scene attributes:', [a for a in dir(scene) if 'node' in a.lower() or 'compositor' in a.lower()])
print('use_nodes:', scene.use_nodes)

# Try to access compositor
try:
    print('scene.node_tree:', scene.node_tree)
except Exception as e:
    print(f'scene.node_tree error: {e}')

# Check view layer
vl = bpy.context.view_layer
print('View layer attributes:', [a for a in dir(vl) if 'node' in a.lower() or 'compositor' in a.lower()])

# Try enabling compositor
scene.use_nodes = True
print('After enabling use_nodes:')
print('use_nodes:', scene.use_nodes)

# Try to get node tree
try:
    print('scene.node_tree:', scene.node_tree)
except Exception as e:
    print(f'scene.node_tree error: {e}')

# Check if there's a compositor attribute
for attr in dir(scene):
    if 'compositor' in attr.lower() or 'node' in attr.lower():
        print(f'scene.{attr}:', getattr(scene, attr, 'N/A'))
