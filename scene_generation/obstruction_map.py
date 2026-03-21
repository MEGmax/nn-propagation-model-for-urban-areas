from __future__ import annotations
 
import json
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
SCENES_ROOT = Path("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes")
 
# These must match what was used in load_sionna_scene.py
RADIO_MAP_CELL_SIZE_M = (0.15, 0.15)
RADIO_MAP_CENTER_M    = (0.0, 0.0, 1.5)   # (cx, cy, rx_height)
RADIO_MAP_SIZE_M      = (16.0, 16.0)
 
# Default TX position — overridden per-scene if tx_metadata.json exists
DEFAULT_TX_WORLD = (0.0, 0.0, 3.0)        # (x, y, z) in metres
 
# =============================================================================
# CORE ALGORITHM
# =============================================================================
 
def generate_obstruction_map(
    elevation_map: np.ndarray,
    tx_pixel: tuple[int, int],
    tx_height_m: float,
    rx_height_m: float,
) -> np.ndarray:
    """
    For every cell (i, j), march along the straight line to the TX cell and
    count how many intermediate cells have a building that intersects the ray.
 
    Parameters
    ----------
    elevation_map : (H, W) float32
        Height of the tallest surface at each cell (0 = ground / no building).
    tx_pixel : (row, col)
        Grid index of the transmitter.
    tx_height_m : float
        Physical height of the TX antenna in metres.
    rx_height_m : float
        Physical height of the RX plane in metres.
 
    Returns
    -------
    obstruction_map : (H, W) int32
        Number of obstructing cells between each cell and the TX.
        0 = clear line-of-sight.  TX cell itself is 0.
    """
    H, W = elevation_map.shape
    obstruction_map = np.zeros((H, W), dtype=np.int32)
 
    tx_row, tx_col = tx_pixel
 
    for i in range(H):
        for j in range(W):
            if i == tx_row and j == tx_col:
                continue  # TX cell — no obstruction by definition
 
            d_row = tx_row - i
            d_col = tx_col - j
            n_steps = max(abs(d_row), abs(d_col))
 
            # Fractional distance along the ray (0 = RX cell, 1 = TX cell)
            ts = np.linspace(0.0, 1.0, n_steps + 1)
 
            sample_rows = np.round(i + ts * d_row).astype(int)
            sample_cols = np.round(j + ts * d_col).astype(int)
 
            # Ray height at each sample — linear from rx_height to tx_height
            ray_heights = rx_height_m + ts * (tx_height_m - rx_height_m)
 
            # Keep only valid interior cells (exclude RX itself and TX)
            valid = (
                (sample_rows >= 0) & (sample_rows < H) &
                (sample_cols >= 0) & (sample_cols < W)
            )
            interior = valid.copy()
            interior[0]  = False  # RX cell
            interior[-1] = False  # TX cell
 
            s_rows = sample_rows[interior]
            s_cols = sample_cols[interior]
            r_heights = ray_heights[interior]
 
            building_heights = elevation_map[s_rows, s_cols]
            is_obstructed = building_heights > r_heights  # bool array
 
            # Count building *crossings* — only the leading edge of each
            # obstructed run counts, so a ray passing through 20 cells of the
            # same building scores 1, not 20.
            # A new obstruction starts where is_obstructed goes False→True.
            if len(is_obstructed) == 0:
                obstruction_map[i, j] = 0
            else:
                # Prepend False so the first cell can trigger a rising edge
                crossings = np.diff(
                    np.concatenate([[False], is_obstructed]).astype(int)
                )
                obstruction_map[i, j] = int(np.sum(crossings == 1))
 
    return obstruction_map
 
 
# =============================================================================
# GRID HELPERS
# =============================================================================
 
def world_to_pixel(
    world_x: float,
    world_y: float,
    cx: float, cy: float,
    sx: float, sy: float,
    dx: float, dy: float,
    W: int,    H: int,
) -> tuple[int, int]:
    """Convert a world-space (x, y) position to the nearest grid pixel."""
    first_x = cx - sx / 2 + dx / 2
    first_y = cy - sy / 2 + dy / 2
    col = int(round((world_x - first_x) / dx))
    row = int(round((world_y - first_y) / dy))
    col = int(np.clip(col, 0, W - 1))
    row = int(np.clip(row, 0, H - 1))
    return row, col
 
 
def load_tx_position(scene_dir: Path) -> tuple[float, float, float]:
    """
    Read TX world position from tx_metadata.json if present,
    otherwise return the module-level default.
    """
    meta_path = scene_dir / "tx_metadata.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        pos = meta.get("tx_position", DEFAULT_TX_WORLD)
        return float(pos[0]), float(pos[1]), float(pos[2])
    return DEFAULT_TX_WORLD
 
 
# =============================================================================
# PER-SCENE PROCESSING
# =============================================================================
 
def process_scene(scene_dir: Path) -> None:
    scene_name = scene_dir.name
    elevation_path = scene_dir / f"elevation_map_{scene_name}.npy"
 
    if not elevation_path.exists():
        print(f"  ⚠  Skipping {scene_name}: elevation_map not found at {elevation_path}")
        return
 
    print(f"Processing {scene_name}...")
 
    elevation_map = np.load(elevation_path)          # (H, W) float32
    H, W = elevation_map.shape
 
    # Grid parameters
    cx, cy    = RADIO_MAP_CENTER_M[0],    RADIO_MAP_CENTER_M[1]
    rx_height = RADIO_MAP_CENTER_M[2]
    sx, sy    = RADIO_MAP_SIZE_M[0],      RADIO_MAP_SIZE_M[1]
    dx, dy    = RADIO_MAP_CELL_SIZE_M[0], RADIO_MAP_CELL_SIZE_M[1]
 
    # TX position
    tx_x, tx_y, tx_z = load_tx_position(scene_dir)
    tx_row, tx_col = world_to_pixel(tx_x, tx_y, cx, cy, sx, sy, dx, dy, W, H)
    print(f"  TX world=({tx_x:.2f}, {tx_y:.2f}, {tx_z:.2f})  →  pixel=({tx_row}, {tx_col})")
 
    # Build obstruction map
    obstruction_map = generate_obstruction_map(
        elevation_map,
        tx_pixel=(tx_row, tx_col),
        tx_height_m=tx_z,
        rx_height_m=rx_height,
    )
 
    # Save
    out_npy = scene_dir / f"obstruction_map_{scene_name}.npy"
    np.save(out_npy, obstruction_map.astype(np.int32))
    print(f"  Saved → {out_npy}")
 
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    im0 = axes[0].imshow(elevation_map, cmap="gray", origin="upper")
    axes[0].plot(tx_col, tx_row, "r*", markersize=10, label="TX")
    axes[0].legend()
    axes[0].set_title(f"Elevation — {scene_name}")
    axes[0].set_xlabel("X (cells)")
    axes[0].set_ylabel("Y (cells)")
    plt.colorbar(im0, ax=axes[0]).set_label("Height (m)")
 
    im1 = axes[1].imshow(obstruction_map, cmap="hot_r", origin="upper")
    axes[1].plot(tx_col, tx_row, "b*", markersize=10, label="TX")
    axes[1].legend()
    axes[1].set_title(f"Obstruction Map — {scene_name}")
    axes[1].set_xlabel("X (cells)")
    axes[1].set_ylabel("Y (cells)")
    plt.colorbar(im1, ax=axes[1]).set_label("# Obstructions")
 
    plt.tight_layout()
    out_png = scene_dir / f"obstruction_plot_{scene_name}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_png}")
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main() -> None:
    if not SCENES_ROOT.exists():
        raise FileNotFoundError(f"Scenes root not found: {SCENES_ROOT}")
 
    scene_dirs = sorted([
        p for p in SCENES_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("scene")
    ])
 
    if not scene_dirs:
        raise FileNotFoundError(f"No scene* directories found in {SCENES_ROOT}")
 
    print(f"Found {len(scene_dirs)} scene(s) in {SCENES_ROOT}\n")
 
    for scene_dir in scene_dirs:
        process_scene(scene_dir)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 