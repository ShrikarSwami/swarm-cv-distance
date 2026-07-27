"""
Drone swarm formation presets for the swarm-cv-distance project.

Each formation function conforms to the same interface as
stage1_geometry.multiview_triangulation_test.make_swarm():

    f(n_drones, area_km=5.0, height_range_m=1000.0, seed=42, ...) -> np.ndarray

Returns an (n_drones, 3) array of positions in meters (ENU-style local frame),
with x, y within [-area_km*500, +area_km*500] and z within [0, height_range_m].

The factory function ``get_formation(name)`` returns a callable matching this
interface, suitable for use in the Blender addon's generate operator.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def grid_formation(
    n_drones: int,
    area_km: float = 5.0,
    height_range_m: float = 1000.0,
    seed: int = 42,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> np.ndarray:
    """Generate drone positions on a regular grid.

    The grid is centred at the origin and spans the full area_km extent.
    Rows and columns are auto-calculated from n_drones if not provided;
    when the grid has more cells than drones, trailing cells are left empty.
    Heights are uniformly random across [0, height_range_m].

    Args:
        n_drones: Number of drones.
        area_km: Area size in km (5 km x 5 km).
        height_range_m: Height range in meters.
        seed: Random seed (used only for height jitter).
        rows: Number of grid rows (auto-calculated if None).
        cols: Number of grid columns (auto-calculated if None).

    Returns:
        (n_drones, 3) array of positions in meters.
    """
    rng = np.random.default_rng(seed)

    if rows is None or cols is None:
        cols = int(np.ceil(np.sqrt(n_drones)))
        rows = int(np.ceil(n_drones / cols))

    half_area = area_km * 500.0  # metres

    x = np.linspace(-half_area, half_area, cols)
    y = np.linspace(-half_area, half_area, rows)
    xx, yy = np.meshgrid(x, y)
    all_xy = np.column_stack([xx.ravel(), yy.ravel()])

    # Take exactly n_drones positions (grid may have more cells than drones)
    xy = all_xy[:n_drones]
    z = rng.uniform(0.0, height_range_m, size=(n_drones, 1))

    return np.hstack([xy, z])


def sphere_formation(
    n_drones: int,
    area_km: float = 5.0,
    height_range_m: float = 1000.0,
    seed: int = 42,
    radius_fraction: float = 0.3,
) -> np.ndarray:
    """Generate drone positions on the surface of a sphere.

    Drones are uniformly distributed on the sphere surface (randomised
    Fibonacci-style via spherical coordinates). The sphere is centred at the
    origin; radius is ``radius_fraction * area_km * 500`` metres.  The sphere
    is then shifted vertically so the lowest drone sits at 100 m AGL (never
    on the ground).

    Args:
        n_drones: Number of drones.
        area_km: Area size in km.
        height_range_m: Height range in meters.
        seed: Random seed.
        radius_fraction: Sphere radius as a fraction of the half-area extent.

    Returns:
        (n_drones, 3) array of positions in meters.
    """
    rng = np.random.default_rng(seed)
    half_area = area_km * 500.0
    radius = half_area * radius_fraction

    # Uniform random directions on the unit sphere
    theta = rng.uniform(0, 2 * np.pi, n_drones)
    phi = np.arccos(2 * rng.uniform(0, 1, n_drones) - 1)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Shift so lowest drone is at 100 m AGL
    z = z - z.min() + 100.0

    # Clamp: if the sphere's vertical extent exceeds height_range_m, rescale
    # z (but not x, y) so everything fits, then re-centre with 100 m floor.
    if z.max() > height_range_m:
        z = z * (height_range_m - 100.0) / z.max()
        z = z - z.min() + 100.0

    return np.column_stack([x, y, z])


def herringbone_formation(
    n_drones: int,
    area_km: float = 5.0,
    height_range_m: float = 1000.0,
    seed: int = 42,
    spacing: float = 200.0,
) -> np.ndarray:
    """Generate drone positions in a herringbone (staggered-row) pattern.

    Alternating rows are offset by half the spacing, giving the classic
    brick/herringbone layout.  Drones that would fall outside the volume
    extent are skipped.  Heights are uniformly random.

    Args:
        n_drones: Number of drones.
        area_km: Area size in km.
        height_range_m: Height range in meters.
        seed: Random seed (used for height jitter).
        spacing: Distance between adjacent drones in metres.

    Returns:
        (n_drones, 3) array of positions in meters.
    """
    rng = np.random.default_rng(seed)
    half_area = area_km * 500.0

    positions: list[list[float]] = []
    row = 0

    while len(positions) < n_drones:
        offset = (spacing / 2.0) if (row % 2 == 1) else 0.0
        col = 0

        while len(positions) < n_drones:
            x = -half_area + col * spacing + offset
            y = -half_area + row * spacing

            if abs(x) <= half_area and abs(y) <= half_area:
                z = rng.uniform(0.0, height_range_m)
                positions.append([x, y, z])

            col += 1
            if -half_area + col * spacing + offset > half_area:
                break

        row += 1
        if -half_area + row * spacing > half_area:
            break

    return np.array(positions[:n_drones])


def lightshow_formation(
    n_drones: int,
    area_km: float = 5.0,
    height_range_m: float = 1000.0,
    seed: int = 42,
    shape: str = "circle",
) -> np.ndarray:
    """Generate drone positions in light-show-style geometric shapes.

    Supported shapes:

    * ``circle``  -- drones equally spaced on a circle.
    * ``star``    -- alternating inner/outer vertices (5-pointed star layout).
    * ``spiral``  -- Archimedean spiral, height increases with arc length.
    * ``line``    -- straight line along the x-axis.

    All shapes are centred at the origin and scaled to fit within half_area.

    Args:
        n_drones: Number of drones.
        area_km: Area size in km.
        height_range_m: Height range in meters.
        seed: Random seed (unused by most shapes, reserved for consistency).
        shape: One of "circle", "star", "spiral", "line".

    Returns:
        (n_drones, 3) array of positions in meters.
    """
    half_area = area_km * 500.0
    radius = half_area * 0.5

    if shape == "circle":
        angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.full(n_drones, height_range_m / 2.0)

    elif shape == "star":
        # 5-pointed star: outer vertices alternate with inner vertices
        angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
        # Every other drone is at the inner radius
        r = np.where(np.arange(n_drones) % 2 == 0, radius, radius * 0.4)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        z = np.full(n_drones, height_range_m / 2.0)

    elif shape == "spiral":
        # Archimedean spiral: radius grows linearly with angle
        t = np.linspace(0, 4 * np.pi, n_drones)
        r_spiral = radius * t / (4 * np.pi)
        x = r_spiral * np.cos(t)
        y = r_spiral * np.sin(t)
        z = np.linspace(100.0, height_range_m - 100.0, n_drones)

    elif shape == "line":
        x = np.linspace(-half_area * 0.8, half_area * 0.8, n_drones)
        y = np.zeros(n_drones)
        z = np.full(n_drones, height_range_m / 2.0)

    else:
        raise ValueError(
            f"Unknown lightshow shape {shape!r}; "
            f"choose from 'circle', 'star', 'spiral', 'line'"
        )

    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Registry: name -> callable(n_drones, area_km, height_range_m, seed, ...) -> (N,3)
_FORMATIONS: dict[str, Callable] = {
    "grid": grid_formation,
    "sphere": sphere_formation,
    "herringbone": herringbone_formation,
    "lightshow_circle": lambda n, **kw: lightshow_formation(n, shape="circle", **kw),
    "lightshow_star": lambda n, **kw: lightshow_formation(n, shape="star", **kw),
    "lightshow_spiral": lambda n, **kw: lightshow_formation(n, shape="spiral", **kw),
    "lightshow_line": lambda n, **kw: lightshow_formation(n, shape="line", **kw),
}


def get_formation(name: str) -> Optional[Callable]:
    """Return a formation generator callable for *name*, or ``None`` for
    ``"random_cloud"`` (which is handled by the existing ``make_swarm``).

    The returned callable has the same signature as ``make_swarm``:
    ``f(n_drones, area_km=5.0, height_range_m=1000.0, seed=42, ...)``.

    Raises:
        KeyError: if *name* is not a recognised formation.
    """
    if name == "random_cloud":
        return None
    return _FORMATIONS[name]


def available_formations() -> list[str]:
    """Return the list of formation names (excluding ``random_cloud``)."""
    return list(_FORMATIONS.keys())
