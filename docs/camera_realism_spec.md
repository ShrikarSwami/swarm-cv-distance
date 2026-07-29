# Camera Realism Integration Specification

This document specifies how to integrate Layer 4b realism features (atmosphere, motion blur, sensor noise, camera jitter) into the `render_clip.py` pipeline. These features complement Layer 4a (procedural terrain + Nishita sky) and are designed for Blender 5.2+ Cycles rendering on Apple Silicon (MPS).

---

## 1. Feature Matrix & Recommendations

| Feature               | Blender Mechanism                  | Render Cost | Quality | Priority | Integration Status |
|-----------------------|-----------------------------------|-------------|---------|----------|-------------------|
| Atmosphere (mist)     | Compositor Mist Pass + Mix        | +0.1s/frame | MEDIUM  | P1       | Ready             |
| Atmosphere (volume)   | World Volume Scatter              | +5-10s/frame| HIGH    | P1*      | Tested (heavy)    |
| Motion Blur           | Cycles Motion Blur + Shutter      | +0.3s/frame | HIGH    | P1**     | Requires animation |
| Sensor Noise          | Compositor Film Grain             | +0.1s/frame | MEDIUM  | P2       | Ready             |
| Camera Jitter         | Python per-frame offsets          | +0s/frame   | MEDIUM  | P2       | Ready             |

*P1 but deferred due to render cost; use Mist Pass for production
**P1 but requires animated drones; currently static swarm pipeline

---

## 2. Integration Points in `render_clip.py`

### 2.1 Scene Setup Phase (after camera creation, before render loop)

```python
# ---- Atmosphere: Enable Mist Pass ----
scene.view_layers[0].use_pass_mist = True
scene.world.mist_settings.start = 0.0
scene.world.mist_settings.depth = 5000.0  # 5km fade distance
scene.world.mist_settings.falloff = "QUADRATIC"

# ---- Motion Blur: Enable in Cycles ----
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.5   # 1/2 frame = 1/48s at 24fps
scene.cycles.motion_blur_position = "CENTER"  # centered on frame

# ---- Sensor Noise: Compositor nodes added in render loop ----
# (added per-frame in the compositor setup section)

# ---- Camera Jitter: Per-frame offsets applied in render loop ----
# (Python random offsets to camera matrix_world)
```

### 2.2 Compositor Setup Per-Frame (inside render loop)

```python
# Rebuild compositor for this camera
node_group.nodes.clear()
rl = node_group.nodes.new("CompositorNodeRLayers")
out = node_group.nodes.new("CompositorNodeOutputFile")
# ... existing ID pass setup ...

# ---- ADD: Atmosphere via Mist Pass ----
if USE_ATMOSPHERE:
    # Mist pass output from Render Layers
    mist_mix = node_group.nodes.new("CompositorNodeMixRGB")
    mist_mix.blend_type = "MIX"
    mist_mix.inputs["Fac"].default_value = 0.3  # haze strength
    mist_mix.inputs["Color2"].default_value = (0.7, 0.75, 0.85, 1.0)  # sky-blue haze
    # Mist is in Alpha channel of Mist pass? No, it's separate output
    # Actually Mist pass output is a separate socket on RLayers
    node_group.links.new(rl.outputs["Mist"], mist_mix.inputs["Fac"])
    node_group.links.new(rl.outputs["Image"], mist_mix.inputs["Color1"])
    # ... chain to output ...

# ---- ADD: Sensor Noise (Film Grain) ----
if USE_SENSOR_NOISE:
    grain_tex = node_group.nodes.new("CompositorNodeTexNoise")
    grain_tex.inputs["Scale"].default_value = 200.0
    grain_mix = node_group.nodes.new("CompositorNodeMixRGB")
    grain_mix.blend_type = "OVERLAY"
    grain_mix.inputs["Fac"].default_value = 0.08  # subtle grain
    # Wire after mist mix (or before output)
```

### 2.3 Per-Frame Render Loop (for jitter and motion blur)

```python
for frame_idx in range(n_frames):
    # ---- Camera Jitter ----
    if USE_CAMERA_JITTER:
        import numpy as np
        rng = np.random.default_rng(seed + frame_idx * 1000)
        pos_jitter = rng.normal(0, 0.5, 3)  # ±0.5m
        rot_jitter = rng.normal(0, np.radians(0.5), 3)  # ±0.5°
        cam_obj.matrix_world = base_matrix @ Matrix.Translation(pos_jitter)
        cam_obj.rotation_euler = (
            base_rot.x + rot_jitter[0],
            base_rot.y + rot_jitter[1],
            base_rot.z + rot_jitter[2],
        )
        bpy.context.view_layer.update()
    
    # ---- Motion Blur ----
    # Requires drones to be animated (keyframed positions)
    # Current pipeline is static per-frame; for motion blur:
    # 1. Pre-animate all drone positions across frames
    # 2. Enable scene.render.use_motion_blur = True
    # 3. Render with motion blur samples (Cycles handles automatically)
    
    scene.camera = cam_obj
    # ... render frame ...
```

---

## 3. Detailed Feature Specifications

### 3.1 Atmospheric Perspective (Distance Haze)

**Recommended: Compositor Mist Pass (P1)**

```python
# World settings
scene.world.mist_settings.start = 0.0
scene.world.mist_settings.depth = 5000.0  # meters
scene.world.mist_settings.falloff = "QUADRATIC"  # or "LINEAR"
scene.view_layers[0].use_pass_mist = True

# Compositor: mix Mist pass with render
# Mist output is grayscale: 0=near (clear), 1=far (hazy)
# Mix RGB: Image * (1-Mist*strength) + HazeColor * Mist*strength
```

**Alternative: World Volume Scatter (High Quality, High Cost)**

```python
# World node tree
vol = nodes.new("ShaderNodeVolumeScatter")
vol.inputs["Density"].default_value = 0.0001  # subtle
vol.inputs["Color"].default_value = (0.7, 0.75, 0.85, 1.0)
vol.inputs["Anisotropy"].default_value = 0.0
links.new(vol.outputs["Volume"], output.inputs["Volume"])

# Cycles settings for volumes
scene.cycles.volume_step_rate = 0.1
scene.cycles.max_volume_bounces = 2
```

**Performance Comparison (1024×512, 32 spp):**
- Baseline: ~2.5s
- Mist Pass: ~2.6s (+0.1s, +4%)
- Volume Subtle: ~7.2s (+4.7s, +188%)
- Volume Heavy: ~11.1s (+8.6s, +344%)

**Recommendation:** Use Mist Pass for production. Volume only for hero frames.

---

### 3.2 Motion Blur

**Requirements:**
- Drones must be animated (keyframed across frames)
- Cycles motion blur enabled in scene settings

**Settings:**
```python
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.5      # 0.5 = 1/2 frame duration
scene.cycles.motion_blur_position = "CENTER"
scene.cycles.use_motion_blur = True  # Cycles-specific
```

**Shutter Speed Reference (24 fps):**
| Setting | Shutter Time | Visual Effect |
|---------|-------------|---------------|
| 1/120s  | 0.0083s     | Barely visible |
| **1/60s**  | **0.0167s** | **Recommended: visible but not overwhelming** |
| 1/30s   | 0.0333s     | Strong blur, may obscure detail |
| 1/20s   | 0.05s       | Heavy blur, drone shapes lost |

**Note:** Current pipeline regenerates drones per-frame (`bpy.ops.mesh.primitive_cube_add`). For motion blur, drones must be **pre-animated** (keyframed positions across frames). This requires pipeline refactor from "rebuild per frame" to "animate once, render loop".

---

### 3.3 Sensor Noise / Film Grain

**Recommended: Compositor-based Film Grain (P2)**

```python
# In compositor, after all other processing but before output
noise = nodes.new("CompositorNodeTexNoise")
noise.inputs["Scale"].default_value = 200.0
noise.inputs["Detail"].default_value = 1.0
noise.inputs["Distortion"].default_value = 0.0

mix = nodes.new("CompositorNodeMixRGB")
mix.blend_type = "OVERLAY"
mix.inputs["Fac"].default_value = 0.08  # 8% grain strength

links.new(prev_node.outputs[0], mix.inputs["Color1"])
links.new(noise.outputs["Fac"], mix.inputs["Color2"])
links.new(mix.outputs["Image"], out.inputs["Image"])
```

**Alternative Approaches (Tested, Not Recommended):**
- Low samples (4-16 spp): Too noisy, destroys detection signal
- ISO simulation: Blender's physical camera ISO doesn't add visible noise without massive exposure
- Denoised + grain: Extra complexity, minimal benefit over clean+grain

**Performance:** +0.1s/frame (negligible)

---

### 3.4 Camera Jitter

**Implementation: Python per-frame offsets (P2)**

```python
import numpy as np

# Pre-compute jitter sequence for reproducibility
rng = np.random.default_rng(SEED + CAMERA_JITTER_OFFSET)
n_frames = cfg.get("n_frames", 1)
jitter_positions = rng.normal(0, 0.5, (n_frames, 3))  # meters
jitter_rotations = rng.normal(0, np.radians(0.5), (n_frames, 3))  # radians

# In render loop:
for frame_idx in range(n_frames):
    # Apply jitter to camera matrix
    jitter_mat = Matrix.Translation(jitter_positions[frame_idx])
    jitter_rot = Euler((
        jitter_rotations[frame_idx][0],
        jitter_rotations[frame_idx][1],
        jitter_rotations[frame_idx][2]
    )).to_matrix().to_4x4()
    
    cam_obj.matrix_world = base_matrix @ jitter_mat @ jitter_rot
    bpy.context.view_layer.update()
    
    # Render frame...
```

**Jitter Profiles Tested:**
| Profile | Position σ | Rotation σ | Use Case |
|---------|-----------|------------|----------|
| Subtle | 0.3m | 0.3° | Production |
| Standard | 0.5m | 0.5° | Default |
| Heavy | 1.0m | 1.0° | Stress test |

**Performance:** +0s/frame (just matrix math)

---

## 4. Configuration Flags for `render_clip.py`

Add to config JSON:

```json
{
  "clip_name": "desert_001",
  "environment": "desert",
  "weather": "clear",
  
  // NEW: Camera realism flags
  "use_atmosphere": true,
  "atmosphere_strength": 0.3,
  "atmosphere_color": [0.7, 0.75, 0.85, 1.0],
  "use_mist_pass": true,
  
  "use_motion_blur": false,  // requires animated drones
  "motion_blur_shutter": 0.5,
  
  "use_sensor_noise": true,
  "grain_strength": 0.08,
  "grain_scale": 200.0,
  
  "use_camera_jitter": true,
  "jitter_position_std": 0.5,
  "jitter_rotation_std_deg": 0.5,
  
  // Existing settings...
  "n_frames": 20,
  "fps": 10,
  ...
}
```

---

## 5. Render Performance Estimates

| Configuration | s/frame (1024×512, 32 spp) | Notes |
|---------------|---------------------------|-------|
| Baseline (terrain + Nishita) | ~3.0s | Current baseline |
| + Mist Pass atmosphere | ~3.1s | +0.1s |
| + Sensor noise (grain) | ~3.1s | +0.1s |
| + Camera jitter | ~3.1s | +0s |
| + Volume atmosphere | ~10s | +7s (NOT recommended) |
| + Motion blur | ~3.3s | +0.3s (needs animation) |
| **Production (mist + grain + jitter)** | **~3.2s** | **+7% overhead** |

---

## 6. Integration Checklist

- [ ] Add config flags to `render_clip.py` argument parsing
- [ ] Import `numpy` at top (already used for camera positions)
- [ ] Add Mist Pass enable in scene setup
- [ ] Add compositor nodes for atmosphere + grain
- [ ] Add camera jitter computation in render loop
- [ ] Add motion blur settings (deferred: requires animation refactor)
- [ ] Test 8-frame matrix: desert/forest × clear/dusk × up/down
- [ ] Verify renders pass visual inspection (no artifacts)
- [ ] Benchmark full clip render time
- [ ] Update smoke test to exercise new flags

---

## 7. Files to Modify

1. **`render_clip.py`** — Main integration (add config flags, scene setup, compositor, render loop)
2. **`smoke_test.py`** — Add configs for testing new flags
3. **`blender_addon/environments.py`** — Keep for now (terrain integration), may deprecate later
4. **`blender_addon/hdri.py`** — Keep for backward compatibility, mark deprecated
5. **`blender_addon/terrain.py`** — **NEW** (already created)
6. **`blender_addon/nishita_sky.py`** — **NEW** (already created)

---

## 8. Notes & Constraints

- **Apple Silicon MPS**: All rendering on CPU (Cycles Metal backend). No CUDA.
- **Blender 5.2**: Compositor uses `scene.compositing_node_group` (NodeGroup), not `scene.node_tree`
- **Reproducibility**: All jitter/noise seeds derived from config `seed` for deterministic renders
- **Detection Compatibility**: Grain/jitter must not break YOLOv8 detection. Keep grain ≤ 0.1 Fac, jitter ≤ 1px projected
- **Backward Compatibility**: New flags default to `false` to not break existing configs

---

*Document generated from Subagent C research. All test outputs available in `dataset_smoke_test/camera_realism/` for visual verification.*