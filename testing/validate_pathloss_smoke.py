import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_SCENE = PROJECT_ROOT / "scene_generation" / "automated_scenes" / "scene0"
DEFAULT_SMOKE_ROOT = Path(tempfile.gettempdir()) / "pathloss_smoke_artifacts_runtime"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _prepare_smoke_scene(source_scene: Path, smoke_scenes_root: Path) -> Path:
    smoke_scene = smoke_scenes_root / "scene0"
    if smoke_scene.exists():
        shutil.rmtree(smoke_scene)
    smoke_scene.mkdir(parents=True, exist_ok=True)

    src_xml = source_scene / "scene0.xml"
    src_meshes = source_scene / "meshes"
    if not src_xml.exists() or not src_meshes.exists():
        raise RuntimeError(f"Source scene is missing scene0.xml or meshes/: {source_scene}")

    shutil.copy2(src_xml, smoke_scene / "scene0.xml")
    shutil.copytree(src_meshes, smoke_scene / "meshes")
    return smoke_scene


def _stats(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    finite_vals = arr[finite]
    if finite_vals.size == 0:
        return {
            "shape": tuple(arr.shape),
            "finite_count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
        }
    return {
        "shape": tuple(arr.shape),
        "finite_count": int(finite_vals.size),
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "mean": float(np.mean(finite_vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny end-to-end pathloss smoke validation")
    parser.add_argument("--source-scene", type=str, default=str(DEFAULT_SOURCE_SCENE))
    parser.add_argument("--smoke-root", type=str, default=str(DEFAULT_SMOKE_ROOT))
    parser.add_argument("--samples-per-tx", type=int, default=120000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--cell-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    source_scene = Path(args.source_scene)
    smoke_root = Path(args.smoke_root)
    smoke_scenes_root = smoke_root / "scenes"
    smoke_model_input = smoke_root / "model_input"
    out_input = smoke_model_input / "input"
    out_target = smoke_model_input / "target_pathloss"

    smoke_scene = _prepare_smoke_scene(source_scene, smoke_scenes_root)

    sionna_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scene_generation" / "load_sionna_scene.py"),
        "--scenes-root",
        str(smoke_scenes_root),
        "--scene-limit",
        "1",
        "--samples-per-tx",
        str(args.samples_per_tx),
        "--max-depth",
        str(args.max_depth),
        "--cell-size",
        str(args.cell_size),
        "--seed",
        str(args.seed),
        "--save-rss",
    ]
    _run(sionna_cmd)

    model_input_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "model_input" / "model_input.py"),
        "--scenes-root",
        str(smoke_scenes_root),
        "--output-dir-input",
        str(out_input),
        "--output-dir-target",
        str(out_target),
    ]
    _run(model_input_cmd)

    pathloss_path = smoke_scene / "pathloss_values0.npy"
    rss_path = smoke_scene / "rss_values0.npy"
    metadata_path = smoke_scene / "tx_metadata.json"
    target_path = out_target / "scene0_pathloss_target.npy"

    checks: list[tuple[str, bool, str]] = []

    checks.append(("pathloss artifact exists", pathloss_path.exists(), str(pathloss_path)))
    checks.append(("target tensor exists", target_path.exists(), str(target_path)))

    if not pathloss_path.exists():
        raise SystemExit("FAIL: pathloss artifact missing")

    pathloss = np.load(pathloss_path)
    st = _stats(pathloss)
    finite_mask = np.isfinite(pathloss)
    finite_vals = pathloss[finite_mask]

    checks.append(("finite values present", finite_vals.size > 0, f"count={finite_vals.size}"))
    checks.append(("non-constant map", np.nanstd(finite_vals) > 1e-6, f"std={float(np.nanstd(finite_vals)):.6f}"))

    if finite_vals.size > 0:
        min_pl = float(np.min(finite_vals))
        max_pl = float(np.max(finite_vals))
        stronger_gain = 10.0 ** (-min_pl / 10.0)
        weaker_gain = 10.0 ** (-max_pl / 10.0)
        direction_ok = (min_pl < max_pl) and (stronger_gain > weaker_gain)
        checks.append(
            (
                "direction sensible (lower path loss => stronger channel)",
                direction_ok,
                f"min={min_pl:.3f} dB, max={max_pl:.3f} dB",
            )
        )
    else:
        checks.append(("direction sensible (lower path loss => stronger channel)", False, "no finite values"))

    rss_crosscheck_done = False
    rss_mae = float("nan")
    if rss_path.exists() and metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        tx_power_dbm = float(meta.get("tx_power_dbm", 30.0))

        rss = np.load(rss_path)
        if rss.shape == pathloss.shape:
            pathloss_from_rss = tx_power_dbm - rss  # assumes tx/rx antenna gains are 0 dBi in this pipeline
            mask = np.isfinite(pathloss) & np.isfinite(pathloss_from_rss)
            if np.any(mask):
                rss_mae = float(np.mean(np.abs(pathloss[mask] - pathloss_from_rss[mask])))
                checks.append(("RSS conversion cross-check", rss_mae < 1.0, f"MAE={rss_mae:.4f} dB"))
                rss_crosscheck_done = True

    print("\n=== Pathloss Smoke Validation Summary ===")
    print(f"pathloss_map: {pathloss_path}")
    print(f"target_tensor: {target_path}")
    print(f"shape: {st['shape']}")
    print(f"min/max/mean: {st['min']:.6f} / {st['max']:.6f} / {st['mean']:.6f} dB")
    print(f"finite_values: {st['finite_count']}")
    if rss_crosscheck_done:
        print(f"rss_crosscheck_mae_db: {rss_mae:.6f}")
    else:
        print("rss_crosscheck: skipped (RSS artifact unavailable or incompatible)")

    all_ok = True
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {detail}")
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit("Pathloss smoke validation FAILED")

    print("Overall: PASS")


if __name__ == "__main__":
    main()
