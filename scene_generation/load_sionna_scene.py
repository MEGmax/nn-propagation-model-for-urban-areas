from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.path import Path as MplPath
from sionna.rt import Camera, PlanarArray, RadioMapSolver, Transmitter, load_scene

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SCENE_DIR = BASE_DIR / "automated_scenes"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.toml"

DEFAULT_FREQUENCY_HZ = 3_500_000_000
RADIO_MAP_CELL_SIZE_M = (0.15, 0.15)
RADIO_MAP_CENTER_M = (0.0, 0.0, 1.5)
RADIO_MAP_SIZE_M = (16.0, 16.0)
RADIO_MAP_MAX_DEPTH = 32
RADIO_MAP_SAMPLES_PER_TX = 10**8

H_MAX = 20.0  # metres — used only for the PNG visualisation


def load_frequency_hz() -> float:
    if not CONFIG_PATH.exists():
        return float(DEFAULT_FREQUENCY_HZ)
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    return float(config.get("Frequency", {}).get("frequency", DEFAULT_FREQUENCY_HZ))


def iter_scene_dirs() -> list[Path]:
    return sorted(
        [
            path
            for path in SCENE_DIR.iterdir()
            if path.is_dir() and (path / f"{path.name}.xml").exists()
        ]
    )


def _to_numpy(array_like):
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def _to_scalar(value) -> float:
    array = _to_numpy(value).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected scalar-like value")
    return float(array[0])

def _elevation_via_mitsuba(scene, grid_xy: np.ndarray) -> np.ndarray | None:
    """
    Cast one downward ray per cell using Mitsuba (Sionna's renderer backend).

    `grid_xy` has shape (H, W, 2) — the (x, y) world-space centre of every
    radio-map cell.  Returns a (H, W) float32 array of surface heights, or
    None if Mitsuba ray-casting is unavailable on this Sionna build.
    """
    try:
        import mitsuba as mi  # noqa: F401 — check availability first
    except ImportError:
        return None

    try:
        mi_scene = scene.mi_scene  # Sionna ≥ 0.17 exposes the Mitsuba scene
    except AttributeError:
        return None

    H, W, _ = grid_xy.shape
    RAY_START_Z = 500.0  # well above any building

    # Flatten to (N, 3) for a single vectorised Mitsuba call
    N = H * W
    xy_flat = grid_xy.reshape(N, 2)

    origins = np.column_stack(
        [xy_flat, np.full(N, RAY_START_Z, dtype=np.float32)]
    ).astype(np.float32)
    directions = np.tile(
        np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (N, 1)
    )

    # Build Mitsuba Ray3f and call scene.ray_intersect
    import mitsuba as mi

    rays = mi.Ray3f(o=mi.Point3f(origins.T), d=mi.Vector3f(directions.T))
    si = mi_scene.ray_intersect(rays)

    # si.is_valid() is a boolean mask; si.p.z gives the hit z-coordinate
    valid = np.array(si.is_valid()).reshape(N)
    hit_z = np.array(si.p.z).reshape(N)

    elevation = np.where(valid, hit_z, 0.0).reshape(H, W).astype(np.float32)
    print("Elevation map built via Mitsuba ray casting.")
    return elevation

def _load_all_buildings(meshes_dir: Path) -> trimesh.Trimesh | None:
    ply_files = sorted(meshes_dir.glob("Building_*.ply"))
    if not ply_files:
        return None
    meshes = [trimesh.load(str(p), force="mesh") for p in ply_files]
    meshes = [m for m in meshes if not m.is_empty]
    return trimesh.util.concatenate(meshes) if meshes else None


def _elevation_via_mesh_rasterisation(
    meshes_dir: Path,
    grid_xy: np.ndarray,
) -> np.ndarray | None:
    """
    Rasterise building meshes directly onto the radio-map grid.

    For each triangle we find which grid cells fall inside it and assign the
    triangle's maximum z-value — identical logic to the original script, but
    the grid is now driven entirely by `grid_xy` (derived from the radio map)
    so the two arrays are pixel-perfect aligned.
    """
    mesh = _load_all_buildings(meshes_dir)
    if mesh is None:
        print("  ⚠  No Building_*.ply files found — elevation map will be all zeros.")
        return None

    H, W, _ = grid_xy.shape
    elevation = np.zeros((H, W), dtype=np.float32)

    vertices = mesh.vertices  # (V, 3)

    # Pre-compute cell spacing from the grid itself (works even if non-uniform)
    # grid_xy[i, j] is the world-space centre of cell (i, j)
    x_coords = grid_xy[0, :, 0]   # shape (W,)
    y_coords = grid_xy[:, 0, 1]   # shape (H,)

    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]

    # Map world coordinates → fractional pixel indices
    # (matches the grid exactly because we derived the extents from it)
    def world_to_pixel(wx, wy):
        px = (wx - x_min) / (x_max - x_min) * (W - 1) if W > 1 else 0.0
        py = (wy - y_min) / (y_max - y_min) * (H - 1) if H > 1 else 0.0
        return px, py

    for face in mesh.faces:
        v0, v1, v2 = vertices[face]

        p0 = world_to_pixel(v0[0], v0[1])
        p1 = world_to_pixel(v1[0], v1[1])
        p2 = world_to_pixel(v2[0], v2[1])

        tri_pix = np.array([p0, p1, p2])
        tri_z = max(v0[2], v1[2], v2[2])

        ix_min = max(0, int(np.floor(tri_pix[:, 0].min())))
        ix_max = min(W - 1, int(np.ceil(tri_pix[:, 0].max())))
        iy_min = max(0, int(np.floor(tri_pix[:, 1].min())))
        iy_max = min(H - 1, int(np.ceil(tri_pix[:, 1].max())))

        path = MplPath(tri_pix)

        for iy in range(iy_min, iy_max + 1):
            for ix in range(ix_min, ix_max + 1):
                if path.contains_point((ix + 0.5, iy + 0.5)):
                    elevation[iy, ix] = max(elevation[iy, ix], tri_z)

    print("Elevation map built via mesh rasterisation (fallback).")
    return elevation


def generate_elevation_map(
    scene,
    radio_map,
    meshes_dir: Path | None = None,
) -> np.ndarray:
    """
    Return a (H, W) float32 elevation array aligned pixel-for-pixel with
    `radio_map`.

    Strategy:
      1. Derive the exact (x, y) world coordinates of every radio-map cell
         from the radio map's own metadata — this is the single source of
         truth for the grid, guaranteeing alignment.
      2. Try Mitsuba ray casting (zero-copy, vectorised, most accurate).
      3. Fall back to mesh rasterisation on the same grid.
      4. If neither works, return an all-zero array with a warning.
    """
    # ------------------------------------------------------------------
    # Step 1: reconstruct the exact radio-map grid
    # ------------------------------------------------------------------
    # radio_map.cell_size  → (cell_x, cell_y) in metres
    # radio_map.center     → (cx, cy, cz)
    # radio_map.size       → (size_x, size_y) in metres
    # All three are guaranteed to be present on any Sionna RadioMap object.

    # radio_map properties may be tuples, lists, 0-d tensors, or 1-d arrays
    # depending on Sionna version — flatten everything to plain Python floats
    def _extract(val, idx):
        v = val[idx] if hasattr(val, "__len__") else val
        return float(np.asarray(v).reshape(-1)[0])

    # Fall back to the module-level constants if the radio_map attributes are
    # not available (older Sionna builds may not expose them)
    try:
        raw_cell = radio_map.cell_size
        raw_center = radio_map.center
        raw_size = radio_map.size
    except AttributeError:
        raw_cell = RADIO_MAP_CELL_SIZE_M
        raw_center = RADIO_MAP_CENTER_M
        raw_size = RADIO_MAP_SIZE_M

    dx = _extract(raw_cell,   0)
    dy = _extract(raw_cell,   1) if hasattr(raw_cell, "__len__") and len(raw_cell) > 1 else dx
    cx = _extract(raw_center, 0)
    cy = _extract(raw_center, 1)
    sx = _extract(raw_size,   0)
    sy = _extract(raw_size,   1) if hasattr(raw_size, "__len__") and len(raw_size) > 1 else sx

    # Number of cells matches the saved path-loss array shape
    path_gain  = _to_numpy(radio_map.path_gain)  # (1, H, W) or (H, W)
    path_gain  = path_gain.squeeze()
    H, W       = path_gain.shape

    # Cell centres in world space
    # Sionna places the first cell at (center - size/2 + cell_size/2)
    xs = np.linspace(cx - sx / 2 + dx / 2, cx + sx / 2 - dx / 2, W)
    ys = np.linspace(cy - sy / 2 + dy / 2, cy + sy / 2 - dy / 2, H)
    grid_x, grid_y = np.meshgrid(xs, ys)           # both (H, W)
    grid_xy = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)

    print(
        f"Radio-map grid: H={H}, W={W}, "
        f"x=[{xs[0]:.2f}, {xs[-1]:.2f}], y=[{ys[0]:.2f}, {ys[-1]:.2f}]"
    )

    # ------------------------------------------------------------------
    # Step 2: try Mitsuba
    # ------------------------------------------------------------------
    elevation = _elevation_via_mitsuba(scene, grid_xy)
    if elevation is not None:
        return elevation

    # ------------------------------------------------------------------
    # Step 3: mesh rasterisation fallback
    # ------------------------------------------------------------------
    if meshes_dir is not None and meshes_dir.is_dir():
        elevation = _elevation_via_mesh_rasterisation(meshes_dir, grid_xy)
        if elevation is not None:
            return elevation

    # ------------------------------------------------------------------
    # Step 4: last resort — zeros
    # ------------------------------------------------------------------
    print("  ⚠  Could not generate elevation map; returning zeros.")
    return np.zeros((H, W), dtype=np.float32)


def render_scene(scene_dir: Path, frequency_hz: float) -> None:
    scene_name = scene_dir.name
    scene_xml = scene_dir / f"{scene_name}.xml"
    meshes_dir = scene_dir / "meshes"
    print(f"Rendering {scene_name} from {scene_xml}...")

    scene = load_scene(scene_xml)
    scene.frequency = int(frequency_hz)
    print(f"Set frequency to {_to_scalar(scene.frequency)} Hz")

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )

    tx = Transmitter("tx", [0, 0, 3.0], [0.0, 0.0, 0.0])
    scene.add(tx)

    camera = Camera(position=[0, 0, 30], look_at=tx.position)
    solver = RadioMapSolver()
    radio_map = solver(
        scene,
        max_depth=RADIO_MAP_MAX_DEPTH,
        samples_per_tx=RADIO_MAP_SAMPLES_PER_TX,
        cell_size=RADIO_MAP_CELL_SIZE_M,
        center=RADIO_MAP_CENTER_M,
        size=RADIO_MAP_SIZE_M,
        orientation=[0, 0, 0],
    )

    scene_metadata = {
        "frequency": _to_scalar(scene.frequency),
        "tx_position": [
            _to_scalar(tx.position.x),
            _to_scalar(tx.position.y),
            _to_scalar(tx.position.z),
        ],
        "tx_orientation": [
            _to_scalar(tx.orientation.x),
            _to_scalar(tx.orientation.y),
            _to_scalar(tx.orientation.z),
        ],
        "radio_map": {
            "cell_size": list(RADIO_MAP_CELL_SIZE_M),
            "center": list(RADIO_MAP_CENTER_M),
            "size": list(RADIO_MAP_SIZE_M),
            "orientation": [0.0, 0.0, 0.0],
        },
    }
    with (scene_dir / "tx_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(scene_metadata, handle, indent=2)

    # --- path loss ---
    path_gain_linear = _to_numpy(radio_map.path_gain)
    path_gain_safe   = np.clip(path_gain_linear, 1e-30, None)
    path_loss_db     = -10.0 * np.log10(path_gain_safe)
    path_loss_db     = np.clip(path_loss_db, 0.0, 200.0)
    np.save(scene_dir / f"pathloss_values_{scene_name}.npy", path_loss_db.astype(np.float32))

    # --- elevation map (grid-aligned) ---
    elevation_map = generate_elevation_map(scene, radio_map, meshes_dir=meshes_dir)
    np.save(
        scene_dir / f"elevation.npy",
        elevation_map.astype(np.float32),
    )

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(path_loss_db.squeeze(), cmap="viridis", origin="upper")
    axes[0].set_title(f"Path Loss — {scene_name}")
    axes[0].set_xlabel("X (cells)")
    axes[0].set_ylabel("Y (cells)")
    plt.colorbar(axes[0].images[0], ax=axes[0]).set_label("dB")

    im = axes[1].imshow(elevation_map, cmap="gray", origin="upper", vmin=0, vmax=H_MAX)
    axes[1].set_title(f"Elevation Map — {scene_name}")
    axes[1].set_xlabel("X (cells)")
    axes[1].set_ylabel("Y (cells)")
    plt.colorbar(im, ax=axes[1]).set_label("Height (m)")

    plt.tight_layout()
    plt.savefig(scene_dir / f"combined_plot_{scene_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Individual path-loss plot (kept for backward compatibility)
    plt.figure()
    plt.imshow(path_loss_db.squeeze(), cmap="viridis")
    plt.colorbar()
    plt.title(f"Path Loss for {scene_name}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(scene_dir / f"pathloss_plot_{scene_name}.png")
    plt.close()

    print(f"Finished {scene_name}")


def main() -> None:
    frequency_hz = load_frequency_hz()
    scene_dirs = iter_scene_dirs()
    if not scene_dirs:
        raise FileNotFoundError(
            f"No scene directories with XML files found in {SCENE_DIR}"
        )

    for scene_dir in scene_dirs:
        render_scene(scene_dir, frequency_hz)


if __name__ == "__main__":
    main()