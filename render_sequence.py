"""
Render a temporal multi-view clip: N frames × M cameras of a flying swarm.

DEPRECATED: This script has an unresolved rendering bug where
bpy.ops.wm.read_factory_settings(use_empty=True) creates materials
whose node trees don't initialize correctly, producing uniform sky
output despite correct scene state. The root cause is that use_empty
creates an empty scene where fresh material node trees lack proper
Blender-internal initialization.

FIX: Use bpy.ops.wm.read_factory_settings() (without use_empty) to
preserve default material infrastructure, then modify existing materials.
See inline_dome_render.py for the working approach.

This file is kept for reference only.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# This script is deprecated. See inline_dome_render.py for working rendering.
print("render_sequence.py is DEPRECATED. Use inline_dome_render.py instead.")
print("Root cause: use_empty=True breaks material node tree initialization.")
print("All scene state was verified identical (field-by-field, id() check).")
