"""Verify what render_clip.py actually does."""
import bpy
import sys
from pathlib import Path

# Add project to path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

# Reset scene
bpy.ops.wm.read_factory_settings()
scene = bpy.context.scene

# Configure render
scene.render.engine = "CYCLES"
scene.cycles.samples = 4
scene.render.resolution_x = 200
scene.render.resolution_y = 200
scene.render.resolution_percentage = 100

# Remove default objects
for name in ["Cube", "Light"]:
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)

print("=" * 70)
print("SCENE STATE BEFORE ENVIRONMENT")
print("=" * 70)
print(f"Objects: {[obj.name for obj in bpy.data.objects]}")

# Import and apply environment
from blender_addon.environments import get_environment
from blender_addon.weather import get_weather

env_preset = get_environment("desert")
weather_preset = get_weather("overcast")

print("\n" + "=" * 70)
print("APPLYING ENVIRONMENT")
print("=" * 70)

env_result = env_preset.apply(scene)
print(f"Environment result: {env_result}")

print("\n" + "=" * 70)
print("SCENE STATE AFTER ENVIRONMENT")
print("=" * 70)
print(f"Objects: {[obj.name for obj in bpy.data.objects]}")
print(f"Meshes: {[mesh.name for mesh in bpy.data.meshes]}")

# Check ground plane
ground = None
for obj in bpy.data.objects:
    if "ground" in obj.name.lower():
        ground = obj
        break

if ground:
    print(f"\nGround plane found: {ground.name}")
    print(f"  Location: {ground.location}")
    bbox = ground.bound_box
    x_size = max(v[0] for v in bbox) - min(v[0] for v in bbox)
    y_size = max(v[1] for v in bbox) - min(v[1] for v in bbox)
    print(f"  Dimensions: {x_size:.0f}m x {y_size:.0f}m")
else:
    print("\nNO GROUND PLANE FOUND!")

# Apply weather
print("\n" + "=" * 70)
print("APPLYING WEATHER")
print("=" * 70)

weather_preset.apply(scene)

print("\n" + "=" * 70)
print("SCENE STATE AFTER WEATHER")
print("=" * 70)
print(f"Objects: {[obj.name for obj in bpy.data.objects]}")

# Check world nodes
world = scene.world
if world and world.use_nodes:
    print(f"\nWorld nodes: {[node.name for node in world.node_tree.nodes]}")
    bg = world.node_tree.nodes.get("Background")
    if bg:
        print(f"Background color: {bg.inputs[0].default_value}")
