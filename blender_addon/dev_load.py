"""
Dev loader for the Swarm Scan addon.

The addon imports shared constants from stage1_geometry/ by repo-relative
path, so it must run from its in-repo location -- Blender's "Install..."
button would copy it into the user addons directory and break that import.

Two ways to load it:

1. From a terminal (opens a normal interactive Blender with the addon live):

     "<path to Blender binary>" --python /path/to/repo/blender_addon/dev_load.py

   Steam install on this machine:
     "$HOME/Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender" \
         --python "$HOME/Projects/swarm-cv-distance/blender_addon/dev_load.py"

2. From inside Blender: open this file in the Scripting workspace's text
   editor and hit Run Script.

Either way, the "Swarm Scan" tab appears in the 3D viewport sidebar (press N).
"""

import os
import sys

_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
if _ADDON_DIR not in sys.path:
    sys.path.insert(0, _ADDON_DIR)

import swarm_scanner

# Re-running this script in the same session should reload, not error.
try:
    swarm_scanner.unregister()
except Exception:
    pass
import importlib

importlib.reload(swarm_scanner)
swarm_scanner.register()
print("Swarm Scan addon registered (View3D sidebar > Swarm Scan tab)")
