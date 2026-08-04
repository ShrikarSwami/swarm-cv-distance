"""T6 — learned multi-view voxel-fusion model.

Owner: model agent. Owned file: `ml/model.py`.

Implements the spec Section 4 architecture against the frozen I/O contract
(`docs/superpowers/specs/ml_contracts.md` SS3):

    forward(views, cameras=None, grid=None) -> np.ndarray float32 (64, 64, 64), >= 0
    extract_positions(heatmap, grid)        -> np.ndarray float32 (K, 3) world metres
    VOXEL_GRID_RES = 64

Architecture (spec SS4, one item per line):
  1. Shared-weight 2D CNN encoder per view.
  2. Back-project per-view features into a shared 64^3 voxel grid over the swarm
     volume using each view's intrinsics (K) and extrinsics (w2c_R, w2c_t).
     Voxel-cell world coordinates use the contract SS3.3 affine map:
     center[axis] + (i + 0.5) * cell - radius_m, cell = 2 * radius_m / 64.
  3. Pool across views at each voxel with mean AND max, concatenated -> fused
     volume. Symmetric pooling gives joint-permutation invariance by construction.
  4. Small 3D CNN decodes the fused volume to an occupancy heatmap (softplus
     head: values strictly > 0, hence >= 0).
  5. extract_positions: 3x3x3 local maxima above a small threshold, clustered,
     each peak refined by soft-argmax over its neighbourhood -> (K, 3) world m.

Permutation invariance (contract SS3.4, Ruling 1): the REAL path
`forward(views, cameras, grid)` is made *bitwise* invariant to any joint
permutation of (view, camera) pairs by sorting the pairs by a canonical
per-camera key (pose-derived) before pooling — mean over a fixed sorted order is
exact, max is order-independent. The pairing is load-bearing: mispairing a view
against a camera changes that view's warped volume and hence the fused output.
The pose-blind control `forward(views)` (cameras=None) sorts views by a content
fingerprint and pools with no pose warp, remaining order-invariant.

Encoder stride (contract SS3.6, HARD): total stride 4 (two stride-2 stages) —
within the <= 8 limit. FIX-01 reduced this from 8->4 so each drone occupies
~2.4 feature pixels at a_max ~9.6px instead of ~1.2, letting the receptive
field resolve neighbouring drones instead of merging them into one plateau.
No 3rd downsampling stage.

Variable view count (contract SS3.2): V in 1..24 handled by the pooling step;
no architecture change between 2 and 8 views.

MPS (contract SS3.7): every op (conv2d/conv3d, groupnorm, adaptive avg pool,
grid_sample, softplus) has a torch MPS implementation. Wave 1 validation runs on
CPU; timing is deferred to a clean measurement window (render campaign occupies
the Metal GPU).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["VOXEL_GRID_RES", "VoxelFusionModel", "forward", "extract_positions"]

# ---------------------------------------------------------------------------
# Frozen constants (contract SS1 table / tests/test_predictions_ml.py)
# ---------------------------------------------------------------------------

VOXEL_GRID_RES = 64
IMAGE_W, IMAGE_H = 1920, 1080


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class VoxelFusionModel(torch.nn.Module):
    """Shared-weight 2D encoder + voxel back-projection + symmetric pooling + 3D decoder.

    Use `forward_volume` for the differentiable path (T6 training) and `forward`
    for the inference numpy path. The module-level `forward` function is a
    lazily-constructed convenience wrapper matching the frozen contract.
    """

    def __init__(self, feat_channels: int = 64):
        super().__init__()
        self.feat_channels = feat_channels
        # FIX-01: encoder stride reduced from 8 -> 4 (two stride-2 stages
        # instead of three). With a_max ~9.6 px each drone now occupies ~2.4
        # feature pixels instead of ~1.2, so the encoder receptive field can
        # resolve neighbouring drones separately instead of merging them into
        # a single broad plateau (the architectural root cause in the
        # FIX_QUEUE diagnosis). The back-projection math is stride-agnostic:
        # `u/self.stride + 0.5` and `wf = 1920/stride` scale together, so the
        # voxel-to-feature-pixel mapping is unchanged in meaning, only at
        # twice the resolution. Contract SS3.6 caps total stride at <= 8;
        # stride 4 sits strictly inside that floor.
        self.stride = 4  # total encoder stride (two stride-2 stages; FIX-01)

        # --- shared-weight 2D encoder (SS4 item 1; stride constraint SS3.6) ---
        # Two stride-2 stages (-> stride 4) + two stride-1 refinement stages.
        # Output (V, feat_channels, 270, 480) at 1080x1920 input.
        enc = []
        cin = 3
        for cout, stride in ((32, 2), (feat_channels, 2),
                             (feat_channels, 1), (feat_channels, 1)):
            enc.append(torch.nn.Conv2d(cin, cout, kernel_size=3, stride=stride,
                                       padding=1, bias=False))
            enc.append(torch.nn.GroupNorm(min(8, cout), cout))
            enc.append(torch.nn.ReLU(inplace=True))
            cin = cout
        self.encoder = torch.nn.Sequential(*enc)

        # --- small 3D CNN decoder (SS4 item 4) ---
        dec = []
        cin3 = 2 * feat_channels  # mean + max pooled, concatenated
        for cout3 in (feat_channels, feat_channels // 2):
            dec.append(torch.nn.Conv3d(cin3, cout3, kernel_size=3, padding=1, bias=False))
            dec.append(torch.nn.GroupNorm(min(8, cout3), cout3))
            dec.append(torch.nn.ReLU(inplace=True))
            cin3 = cout3
        dec.append(torch.nn.Conv3d(cin3, 1, kernel_size=1))
        dec.append(torch.nn.Softplus())  # values >= 0 (SS3.5)
        self.decoder = torch.nn.Sequential(*dec)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def forward_volume(self, views, cameras=None, grid=None):
        """Differentiable forward path.

        views:  list of V float32 (3, 1080, 1920) arrays, or a stacked
                (V, 3, 1080, 1920) float32 tensor.
        cameras: None, or list of V camera dicts (contract SS3.2).
        grid:    None, or {"center": float32 (3,), "radius_m": float}.

        Returns torch.Tensor (1, 1, 64, 64, 64), values >= 0.
        """
        device = next(self.parameters()).device
        x = self._views_to_tensor(views, device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        v = x.shape[0]
        if v < 1 or v > 24:
            raise ValueError("view count must be in 1..24, got %d" % v)

        if cameras is not None:
            if len(cameras) != v:
                raise ValueError("got %d views but %d cameras"
                                 % (v, len(cameras)))

        feats = self.encoder(x)  # (V, C, 270, 480) at stride 4 (FIX-01)

        if cameras is None:
            fused = self._pool_pose_blind(feats)
        else:
            fused = self._pool_backproject(feats, cameras, grid)

        h = self.decoder(fused.unsqueeze(0))  # (1, 1, 64, 64, 64)
        return h

    def forward(self, views, cameras=None, grid=None):
        """Inference entry point: returns np.ndarray float32 (64, 64, 64), >= 0."""
        h = self.forward_volume(views, cameras=cameras, grid=grid)
        return h.detach().cpu().numpy()[0, 0].astype(np.float32)

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool_pose_blind(self, feats):
        """Pose-blind control (SS3.4): no warp, every view treated identically.

        Each view's 2D features are spatially aggregated to (64, 64) and expanded
        along the depth axis; mean+max pooling across views is symmetric, and the
        views are processed in a content-derived canonical order so the result is
        bitwise invariant to reordering.
        """
        order = self._view_order(feats)
        feats = feats[order]
        agg = F.adaptive_avg_pool2d(feats, (VOXEL_GRID_RES, VOXEL_GRID_RES))
        vol = agg.unsqueeze(-1).expand(-1, -1, -1, -1, VOXEL_GRID_RES)
        mean = vol.mean(dim=0)
        mx, _ = vol.max(dim=0)
        return torch.cat([mean, mx], dim=0)

    def _pool_backproject(self, feats, cameras, grid):
        """Back-project per-view features into the shared voxel grid (SS4 item 2)."""
        v = feats.shape[0]
        # Canonical pair order (camera pose key): bitwise joint-permutation
        # invariance, with the (view, camera) pairing still load-bearing.
        order = self._camera_order(cameras)
        feats = feats[order]
        cameras = [cameras[i] for i in order]

        device = feats.device
        res = VOXEL_GRID_RES
        center, radius = self._grid_params(grid)
        cell = 2.0 * radius / res

        # Voxel-cell world coordinates (contract SS3.3 affine map).
        idx = torch.arange(res, device=device, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(idx, idx, idx, indexing="ij")
        centers = (torch.stack([xx, yy, zz], dim=-1) + 0.5) * cell - radius
        centers = centers + torch.as_tensor(center, dtype=torch.float32, device=device)
        xyz = centers.reshape(-1, 3)  # (G, 3)

        K = torch.as_tensor(
            np.stack([np.asarray(c["K"], dtype=np.float64) for c in cameras]),
            dtype=torch.float32, device=device)                       # (V, 3, 3)
        R = torch.as_tensor(
            np.stack([np.asarray(c["w2c_R"], dtype=np.float64) for c in cameras]),
            dtype=torch.float32, device=device)                       # (V, 3, 3)
        t = torch.as_tensor(
            np.stack([np.asarray(c["w2c_t"], dtype=np.float64) for c in cameras]),
            dtype=torch.float32, device=device)                       # (V, 3)

        # cam = R @ X + t  (OpenCV convention);  p = K @ cam.
        cam = torch.einsum("gd,vcd->gvc", xyz, R) + t.unsqueeze(0)    # (G, V, 3)
        p = torch.einsum("gvc,vdc->gvd", cam, K)                      # (G, V, 3)
        depth = cam[..., 2]
        u = p[..., 0] / p[..., 2]
        vp = p[..., 1] / p[..., 2]   # vertical pixel coordinate (v is the view count)

        valid = (depth > 0) & (u >= 0) & (u < IMAGE_W) & (vp >= 0) & (vp < IMAGE_H)

        # Feature-pixel coordinates, then grid_sample normalized coordinates.
        hf, wf = feats.shape[-2], feats.shape[-1]
        nx = 2.0 * (u / self.stride + 0.5) / wf - 1.0
        ny = 2.0 * (vp / self.stride + 0.5) / hf - 1.0
        grid = torch.stack([nx, ny], dim=-1).permute(1, 0, 2)         # (V, G, 2)
        grid = grid.reshape(v, -1, 1, 2)
        # Voxels behind a camera or off-image get no evidence from that view.
        grid[~valid.permute(1, 0)] = -2.0

        sampled = F.grid_sample(feats, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        # sampled: (V, C, G, 1) -> (V, C, 64, 64, 64)
        vol = sampled[..., 0].reshape(v, self.feat_channels, res, res, res)

        # Symmetric pooling (SS4 item 3): mean AND max, concatenated.
        mean = vol.mean(dim=0)
        mx, _ = vol.max(dim=0)
        return torch.cat([mean, mx], dim=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _views_to_tensor(views, device):
        if isinstance(views, torch.Tensor):
            x = views
        elif isinstance(views, np.ndarray):
            x = torch.as_tensor(views)
        else:
            if all(isinstance(v, torch.Tensor) for v in views):
                x = torch.stack(views)
            else:
                x = torch.as_tensor(
                    np.stack([np.asarray(v, dtype=np.float32) for v in views]))
        return x.to(device=device, dtype=torch.float32)

    @staticmethod
    def _grid_params(grid):
        if grid is None:
            raise ValueError(
                "grid is required when cameras are given (contract SS3.3): "
                "{'center': float32 (3,), 'radius_m': float}")
        center = np.asarray(grid["center"], dtype=np.float32)
        radius = float(grid["radius_m"])
        return center, radius

    @staticmethod
    def _camera_order(cameras):
        """Canonical per-camera sort key: pose-derived (w2c_R, w2c_t), so the
        order is a function of the camera alone and invariant to the input
        ordering of the (view, camera) pairs."""
        keys = []
        for c in cameras:
            r = np.asarray(c["w2c_R"], dtype=np.float64).ravel()
            t = np.asarray(c["w2c_t"], dtype=np.float64).ravel()
            keys.append(tuple(float(x) for x in r) + tuple(float(x) for x in t))
        return sorted(range(len(cameras)), key=lambda i: keys[i])

    @staticmethod
    def _view_order(feats):
        """Content-derived canonical order for the pose-blind path."""
        keys = []
        for i in range(feats.shape[0]):
            f = feats[i].detach()  # fingerprint only — no gradient needed
            keys.append((float(f.sum()), float(f[0, 0, 0])))
        return sorted(range(feats.shape[0]), key=lambda i: keys[i])


# ---------------------------------------------------------------------------
# Module-level contract API (frozen SS3.1)
# ---------------------------------------------------------------------------

_MODEL = None
_MODEL_SEED = 0


def forward(views, cameras=None, grid=None):
    """Fused 3D occupancy heatmap. Returns np.ndarray float32 (64, 64, 64), >= 0.

    Contract SS3: views is a list of V float32 (3, 1080, 1920) arrays or a
    stacked (V, 3, 1080, 1920) tensor; cameras is None (pose-blind control) or a
    list of V camera dicts; grid is the back-projection volume (required when
    cameras is given). Joint (view, camera) permutation leaves the output
    unchanged; the pairing is load-bearing; the pose-blind path is order-invariant.
    """
    global _MODEL
    if _MODEL is None:
        torch.manual_seed(_MODEL_SEED)
        _MODEL = VoxelFusionModel()
        _MODEL.eval()
    return _MODEL.forward(views, cameras=cameras, grid=grid)


def extract_positions(heatmap, grid):
    """Local maxima + per-peak soft-argmax -> predicted 3D positions (SS3.5).

    heatmap: (64, 64, 64) float32, values >= 0.
    grid: {"center": float32 (3,), "radius_m": float} — the back-projection
          volume the heatmap lives in.
    Returns np.ndarray float32 (K, 3) world metres; K = number of detected peaks
    (the model's predicted count, feeds count_err).
    """
    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim == 4:
        h = h[0]
    if h.shape != (VOXEL_GRID_RES, VOXEL_GRID_RES, VOXEL_GRID_RES):
        raise ValueError("heatmap must be (64, 64, 64), got %s" % (h.shape,))
    if grid is None:
        raise ValueError("grid is required (contract SS3.3): "
                         "{'center': float32 (3,), 'radius_m': float}")

    center = np.asarray(grid["center"], dtype=np.float64)
    radius = float(grid["radius_m"])
    res = VOXEL_GRID_RES
    cell = 2.0 * radius / res

    # Voxel-cell world coordinates (contract SS3.3 affine map) — (64, 64, 64, 3).
    idx = np.arange(res, dtype=np.float64)
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    centers = center + (np.stack([xx, yy, zz], axis=-1) + 0.5) * cell - radius

    # 3x3x3 local maxima above a small threshold.
    maxima = _local_maxima(h)
    thresh = max(float(h.max()) * 1e-3, 1e-6)
    peaks = np.argwhere(maxima & (h >= thresh))
    if len(peaks) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Value-descending order, then cluster (merge peaks within 1.5 voxels).
    vals = h[maxima & (h >= thresh)]
    order = np.argsort(-vals, kind="stable")
    peaks = peaks[order]
    if len(peaks) > 512:  # hard cap bounds the (quadratic) clustering
        peaks = peaks[:512]
    kept = []
    for p in peaks:
        if all(np.linalg.norm(p - q) > 1.5 for q in kept):
            kept.append(p)
    kept = np.asarray(kept, dtype=np.int64)

    # Per-peak soft-argmax over a 5x5x5 neighbourhood.
    out = np.zeros((len(kept), 3), dtype=np.float32)
    for a, p in enumerate(kept):
        lo = np.maximum(p - 2, 0)
        hi = np.minimum(p + 3, res)
        win = centers[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1, 3)
        w = h[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].ravel().astype(np.float64)
        s = w.sum()
        if s > 0:
            out[a] = (w[:, None] * win).sum(axis=0) / s
        else:
            out[a] = centers[tuple(p)]
    return out


def _local_maxima(h):
    """Boolean mask of 3x3x3 local maxima (value >= all 26 neighbours)."""
    hp = np.pad(h, 1, mode="constant", constant_values=-np.inf)
    mx = hp[1:-1, 1:-1, 1:-1]
    res = VOXEL_GRID_RES
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                mx = np.maximum(mx, hp[1 + di:1 + di + res,
                                       1 + dj:1 + dj + res,
                                       1 + dk:1 + dk + res])
    return h >= mx
