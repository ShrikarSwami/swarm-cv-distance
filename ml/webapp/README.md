# Web Frontend for Reconstruction UI (T9b)

This directory contains the web front-end wrapper around the existing ml/recon_app.py reconstruction UI.

## Purpose

Wrap the existing geometric reconstruction pipeline (T9 - ml/recon_app.py) in a browser-based UI per the original specification. This is a wrapper, not a reimplementation of the core reconstruction logic.

## Agent Ownership

**Agent M** owns this directory exclusively. Files may import from ml/recon_app.py and the frozen pipeline (ml.baseline_adapter, ml.metrics, etc.), but MUST NOT modify them.

## Project Structure

- **backend/**: FastAPI backend (Python)
  - `main.py`: FastAPI application entry point
  - `api.py`: Reconstruction API endpoints
  - `scene_loader.py`: Manifest loading and scene management
  - `models.py`: Pydantic models for request/response
  
- **frontend/**: Three.js web interface (JavaScript/TypeScript)
  - `index.html`: Main HTML page
  - `src/`: Source files
    - `main.js`: Core application logic
    - `scene-selector.js`: Scene selection UI
    - `angle-selector.js`: Angle selection UI
    - `recon-viewer.js`: 3D reconstruction viewer
    - `metrics-panel.js`: Metrics display
  
- **static/`: Static assets
  - `styles/`: CSS
  - `assets/`: Thumbnail previews, etc.

## Requirements

### 1. Scene Selection
- List TEST-split scenes from ~/swarm_ml/manifest.jsonl
- Show seed, drone count, cell (primary/secondary), a_max
- Thumbnail preview of one render per scene

### 2. Angle Selection (Three modes)
- **Exact**: Thumbnail grid of all 24 angles, labelled by tier (ground/level/aerial), click to toggle
- **Random N**: User picks a count, app samples from scene's rendered set
- **Random random**: App picks both count and angles from scene's rendered set
- Angles are selected from the scene's rendered set, not arbitrary uploads

### 3. Reconstruction View
- Side-by-side or overlaid 3D: true positions vs reconstructed
- Orbit/zoom (Three.js OrbitControls)
- Distinguish visually: matched, missed, phantom/false tracks
- Show selected input renders with tier labels

### 4. Metrics Panel (via frozen ml/metrics.py)
- mAP, median position error, count error, precision/recall
- Number of views used and tier composition
- Adjacency F1 at user-selectable d_max (using ml/adjacency_eval.py)

### 5. Framing
- Visible label: "Reconstruction method: geometric (epipolar + DLT)"
- No need for users to understand the method

## API Endpoints

- `GET /api/scenes`: List available test scenes
- `GET /api/scene/{seed}`: Get scene details
- `POST /api/reconstruct`: Run reconstruction with selected scene and angles
- `GET /api/thumbnail/{seed}`: Get scene thumbnail
- `GET /api/cameras/{seed}`: Get camera poses for a scene

## Build Instructions

1. Ensure ~/swarm_ml is present
2. Install Python dependencies: fastapi, uvicorn, numpy, scipy
3. Run backend: `uvicorn ml.webapp.backend.main:app --reload`
4. Run frontend: Serve static files from ml/webapp/frontend
5. Access at: http://localhost:8000

## Acceptance Criteria

1. Start server and select real TEST scene
2. Run all three angle modes
3. Confirm 3D view renders correctly
4. Verify metrics match ml/recon_app.py outputs for same scene/view set
5. Report scene and matching numbers from both paths