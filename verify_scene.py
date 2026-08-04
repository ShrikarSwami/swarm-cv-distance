"""Verify what's actually in the Blender scene during render."""
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
print("SCENE STATE AFTER REMOVING DEFAULTS")
print("=" * 70)
print(f"Objects: {[obj.name for obj in bpy.data.objects]}")
print(f"Meshes: {[mesh.name for mesh in bpy.data.meshes]}")
print(f"Materials: {[mat.name for mat in bpy.data.materials]}")
print()

# Now try to apply environment using the new environments.py
try:
    from blender_addon.environments import get_environment

    print("=" * 70)
    print("APPLYING DESERT ENVIRONMENT")
    print("=" * 70)

    env = get_environment("desert")
    env.apply(scene)

    print(f"Objects after: {[obj.name for obj in bpy.data.objects]}")
    print(f"Meshes after: {[mesh.name for mesh in bpy.data.meshes]}")
    print(f"Materials after: {[mat.name for mat in bpy.data.materials]}")

    # Check if ground plane exists
    ground = None
    for obj in bpy.data.objects:
        if "ground" in obj.name.lower() or "plane" in obj.name.lower():
            ground = obj
            break

    if ground:
        print(f"\nGround plane found: {ground.name}")
        print(f"  Location: {ground.location}")
        print(f"  Scale: {ground.scale}")
        # Check dimensions
        bbox = ground.bound_box
        x_size = max(v[0] for v in bbox) - min(v[0] for v in bbox)
        y_size = max(v[1] for v in bbox) - min(v[1] for v in bbox)
        print(f"  Dimensions: {x_size:.0f}m x {y_size:.0f}m")
    else:
        print("\nNO GROUND PLANE FOUND!")

    # Check world background
    world = scene.world
    if world and world.use_nodes:
        print(f"\nWorld nodes: {[node.name for node in world.node_tree.nodes]}")
    else:
        print("\nNO WORLD NODES!")

except ImportError as e:
    print(f"Failed to import environments: {e}")

# Now test what render_clip.py actually does
print("\n" + "=" * 70)
print("TESTING WHAT render_clip.py ACTUALLY DOES")
print("=" * 70)

# Reset scene
bpy.ops.wm.read_factory_settings()
scene = bpy.context.scene

# Remove default objects
for name in ["Cube", "Light"]:
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)

# Simulate render_clip.py's environment setup
ENV_PRESETS = {
    "desert": {"bg_color": (0.8, 0.7, 0.5, 1.0), "sun_energy": 4.0, "sun_color": (1.0, 0.95, 0.8)},
    "forest": {"bg_color": (0.4, 0.6, 0.4, 1.0), "sun_energy": 3.0, "sun_color": (0.9, 1.0, 0.8)},
    "city":   {"bg_color": (0.5, 0.5, 0.6, 1.0), "sun_energy": 3.5, "sun_color": (0.95, 0.95, 1.0)},
}
WEATHER_PRESETS = {
    "clear":   {"energy_mult": 1.0},
    "overcast": {"energy_mult": 0.5},
    "hazy":    {"energy_mult": 0.7},
}

env_name = "desert"
weather_name = "overcast"

env = ENV_PRESETS[env_name]
weather = WEATHER_PRESETS[weather_name]

print(f"Using environment: {env_name}")
print(f"Using weather: {weather_name}")

# Create light (like render_clip.py does)
light_data = bpy.data.lights.new("Sun", "SUN")
light_data.energy = env["sun_energy"] * weather["energy_mult"]
light_data.color = env["sun_color"][:3]
light_obj = bpy.data.objects.new("Sun", light_data)
light_obj.location = (0, 0, 10000)
bpy.context.collection.objects.link(light_obj)

# Set world background (like render_clip.py does)
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = env["bg_color"]

print(f"\nObjects: {[obj.name for obj in bpy.data.objects]}")
print(f"Sun energy: {light_data.energy} (base={env['sun_energy']}, weather_mult={weather['energy_mult']})")
print(f"World background color: {bg.inputs[0].default_value if bg else 'N/A'}")

# Check if there's a ground plane
ground = None
for obj in bpy.data.objects:
    if "ground" in obj.name.lower() or "plane" in obj.name.lower():
        ground = obj
        break

if ground:
    print(f"Ground plane found: {ground.name}")
else:
    print("NO GROUND PLANE FOUND - render_clip.py doesn't create one!")
