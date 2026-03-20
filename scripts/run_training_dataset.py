"""Run training, backtest, visualization, and optional inference on a
preprocessed multi-scene dataset.

Example:
    python scripts/run_training_dataset.py \
      --input-dir model_input/data/training/input \
      --target-dir model_input/data/training/target \
      --stats-file model_input/data/training/normalization_stats.json \
      --artifacts-root artifacts/training_run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "model_input" / "data" / "training" / "input"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "model_input" / "data" / "training" / "target"
DEFAULT_STATS_FILE = PROJECT_ROOT / "model_input" / "data" / "training" / "normalization_stats.json"
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "training_run"
SCENE_INPUT_PATTERN = re.compile(r"^(scene\d+)_input\.npy$")
STAGE_ORDER = ("train", "backtest", "visualize", "inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training and evaluation on a preprocessed multi-scene dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing scene*_input.npy files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory containing scene*_target.npy files.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=DEFAULT_STATS_FILE,
        help="Normalization statistics JSON file.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory where outputs will be stored.",
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--samples-per-scene", type=int, default=1)
    parser.add_argument("--num-scenes-vis", type=int, default=5)
    parser.add_argument(
        "--inference-scenes",
        nargs="*",
        default=[],
        help="Optional scene names for direct inference, e.g. scene0 scene5 scene19",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter to use for subprocess commands.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def tail_text(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text.strip()
    return text[-max_chars:].strip()


def run_logged_subprocess(
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    cwd: Path,
) -> tuple[int, float]:
    start_time = time.time()
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        header = f"$ {format_command(command)}\n\n"
        stdout_handle.write(header)
        stderr_handle.write(header)
        stdout_handle.flush()
        stderr_handle.flush()

        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )

        duration_seconds = time.time() - start_time
        footer = f"\n[exit_code={completed.returncode} duration_seconds={duration_seconds:.2f}]\n"
        stdout_handle.write(footer)
        stderr_handle.write(footer)

    return completed.returncode, duration_seconds


def discover_scene_names(input_dir: Path, target_dir: Path) -> list[str]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    scene_names: list[str] = []
    for path in sorted(input_dir.glob("scene*_input.npy")):
        match = SCENE_INPUT_PATTERN.fullmatch(path.name)
        if not match:
            continue
        scene_name = match.group(1)
        target_file = target_dir / f"{scene_name}_target.npy"
        if target_file.exists():
            scene_names.append(scene_name)

    scene_names.sort(key=lambda name: int(name.replace("scene", "")))
    return scene_names


def ensure_output_directories(artifacts_root: Path) -> dict[str, Path]:
    paths = {
        "root": artifacts_root,
        "checkpoints": artifacts_root / "checkpoints",
        "backtest": artifacts_root / "backtest",
        "visuals": artifacts_root / "visuals",
        "inference": artifacts_root / "inference",
        "logs": artifacts_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def train_complete(paths: dict[str, Path]) -> bool:
    return (paths["checkpoints"] / "model_final.pt").exists()


def backtest_complete(paths: dict[str, Path]) -> bool:
    return (paths["backtest"] / "backtest_results.json").exists()


def visualize_complete(paths: dict[str, Path]) -> bool:
    return any(paths["visuals"].glob("*.png"))


def inference_complete(paths: dict[str, Path], scene_name: str, timesteps: int) -> bool:
    return (paths["inference"] / f"{scene_name}_prediction_t{timesteps}.npy").exists()


def build_train_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        args.python_executable,
        "models/model.py",
        "--input-dir",
        str(args.input_dir),
        "--target-dir",
        str(args.target_dir),
        "--stats-file",
        str(args.stats_file),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--timesteps",
        str(args.timesteps),
        "--save-every",
        str(args.save_every),
        "--checkpoint-dir",
        str(paths["checkpoints"]),
        "--num-workers",
        str(args.num_workers),
    ]


def build_backtest_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        args.python_executable,
        "backtest/run_backtest.py",
        "--checkpoint",
        str(paths["checkpoints"] / "model_final.pt"),
        "--input-dir",
        str(args.input_dir),
        "--target-dir",
        str(args.target_dir),
        "--stats-file",
        str(args.stats_file),
        "--output-dir",
        str(paths["backtest"]),
        "--batch-size",
        str(args.batch_size),
        "--samples-per-scene",
        str(args.samples_per_scene),
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--timesteps",
        str(args.timesteps),
    ]


def build_visualize_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        args.python_executable,
        "visualize/visualize_rss_maps.py",
        "--checkpoint",
        str(paths["checkpoints"] / "model_final.pt"),
        "--input-dir",
        str(args.input_dir),
        "--target-dir",
        str(args.target_dir),
        "--stats-file",
        str(args.stats_file),
        "--output-dir",
        str(paths["visuals"]),
        "--num-scenes",
        str(args.num_scenes_vis),
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--timesteps",
        str(args.timesteps),
    ]


def build_inference_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    scene_name: str,
) -> list[str]:
    output_name = f"{scene_name}_prediction_t{args.timesteps}"
    return [
        args.python_executable,
        "models/inference.py",
        "--checkpoint",
        str(paths["checkpoints"] / "model_final.pt"),
        "--input",
        str(args.input_dir / f"{scene_name}_input.npy"),
        "--stats-file",
        str(args.stats_file),
        "--output-dir",
        str(paths["inference"]),
        "--output-name",
        output_name,
        "--sampling-steps",
        str(args.diffusion_steps),
        "--timesteps",
        str(args.timesteps),
    ]


def run_stage(
    stage_name: str,
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    completion_check,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    already_complete = completion_check()

    if not force and skip_existing and already_complete:
        write_text(stdout_log, f"Skipping {stage_name}; completion markers already exist.\n")
        write_text(stderr_log, "")
        return {
            "name": stage_name,
            "status": "skipped",
            "completed": True,
            "return_code": None,
            "duration_seconds": 0.0,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "error_message": None,
        }

    return_code, duration_seconds = run_logged_subprocess(
        command=command,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        cwd=PROJECT_ROOT,
    )
    completed = completion_check()

    if return_code != 0:
        return {
            "name": stage_name,
            "status": "failed",
            "completed": completed,
            "return_code": return_code,
            "duration_seconds": duration_seconds,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "error_message": (
                f"{stage_name} failed with exit code {return_code}. "
                f"See {stderr_log}. Tail:\n{tail_text(stderr_log) or tail_text(stdout_log)}"
            ),
        }

    if not completed:
        return {
            "name": stage_name,
            "status": "failed",
            "completed": False,
            "return_code": return_code,
            "duration_seconds": duration_seconds,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "error_message": f"{stage_name} finished but expected outputs were not found.",
        }

    return {
        "name": stage_name,
        "status": "completed",
        "completed": True,
        "return_code": return_code,
        "duration_seconds": duration_seconds,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "error_message": None,
    }


def first_numeric(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_backtest_metrics(path: Path) -> dict[str, float | None]:
    metrics = {
        "rmse": None,
        "mae": None,
        "median_error": None,
        "bias": None,
        "pearson_r": None,
    }
    if not path.exists():
        return metrics

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return metrics

    source: dict[str, Any] = {}
    if isinstance(payload.get("aggregate"), dict):
        source = payload["aggregate"]
    elif isinstance(payload.get("per_scene"), list) and payload["per_scene"]:
        maybe = payload["per_scene"][0]
        if isinstance(maybe, dict):
            source = maybe

    metrics["rmse"] = first_numeric(source, "rmse_db", "rmse_db_mean")
    metrics["mae"] = first_numeric(source, "mae_db", "mae_db_mean")
    metrics["median_error"] = first_numeric(source, "median_error_db", "median_error_db_mean")
    metrics["bias"] = first_numeric(source, "bias_db", "bias_db_mean")
    metrics["pearson_r"] = first_numeric(source, "pearson_correlation", "pearson_correlation_mean")
    return metrics


def write_summary(
    args: argparse.Namespace,
    paths: dict[str, Path],
    scene_names: list[str],
    stage_results: dict[str, Any],
) -> None:
    summary = {
        "dataset": {
            "input_dir": str(args.input_dir),
            "target_dir": str(args.target_dir),
            "stats_file": str(args.stats_file),
            "num_scenes": len(scene_names),
            "scene_names": scene_names,
        },
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timesteps": args.timesteps,
            "save_every": args.save_every,
            "num_workers": args.num_workers,
            "diffusion_steps": args.diffusion_steps,
            "samples_per_scene": args.samples_per_scene,
            "num_scenes_vis": args.num_scenes_vis,
            "inference_scenes": args.inference_scenes,
        },
        "stage_results": stage_results,
        "metrics": load_backtest_metrics(paths["backtest"] / "backtest_results.json"),
    }

    (paths["root"] / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    csv_path = paths["root"] / "run_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "status", "completed", "duration_seconds", "return_code", "error_message"],
        )
        writer.writeheader()
        for stage_name in STAGE_ORDER:
            result = stage_results.get(stage_name)
            if result is None:
                continue
            writer.writerow(
                {
                    "stage": stage_name,
                    "status": result["status"],
                    "completed": result["completed"],
                    "duration_seconds": result["duration_seconds"],
                    "return_code": result["return_code"],
                    "error_message": result["error_message"],
                }
            )


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.target_dir = args.target_dir.resolve()
    args.stats_file = args.stats_file.resolve()
    args.artifacts_root = args.artifacts_root.resolve()

    scene_names = discover_scene_names(args.input_dir, args.target_dir)
    if not scene_names:
        print("No matching scene*_input.npy / scene*_target.npy pairs found.")
        return 1

    paths = ensure_output_directories(args.artifacts_root)

    print(f"Discovered {len(scene_names)} scenes.")
    print(f"Input dir:  {args.input_dir}")
    print(f"Target dir: {args.target_dir}")
    print(f"Stats file: {args.stats_file}")

    stage_results: dict[str, Any] = {}

    stage_specs = [
        (
            "train",
            build_train_command(args, paths),
            lambda: train_complete(paths),
        ),
        (
            "backtest",
            build_backtest_command(args, paths),
            lambda: backtest_complete(paths),
        ),
        (
            "visualize",
            build_visualize_command(args, paths),
            lambda: visualize_complete(paths),
        ),
    ]

    for stage_name, command, completion_check in stage_specs:
        print(f"\n[{stage_name}]")
        print(format_command(command))
        result = run_stage(
            stage_name=stage_name,
            command=command,
            stdout_log=paths["logs"] / f"{stage_name}.stdout.log",
            stderr_log=paths["logs"] / f"{stage_name}.stderr.log",
            completion_check=completion_check,
            force=args.force,
            skip_existing=args.skip_existing,
        )
        stage_results[stage_name] = result

        if result["status"] == "completed":
            print(f"completed in {result['duration_seconds']:.2f}s")
            continue
        if result["status"] == "skipped":
            print("skipped")
            continue

        print("failed")
        write_summary(args, paths, scene_names, stage_results)
        return 1

    inference_results = []
    for scene_name in args.inference_scenes:
        if scene_name not in scene_names:
            inference_results.append(
                {
                    "scene_name": scene_name,
                    "status": "failed",
                    "completed": False,
                    "duration_seconds": 0.0,
                    "return_code": None,
                    "error_message": f"{scene_name} not found in dataset.",
                }
            )
            if args.stop_on_error:
                break
            continue

        command = build_inference_command(args, paths, scene_name)
        print(f"\n[inference:{scene_name}]")
        print(format_command(command))
        result = run_stage(
            stage_name=f"inference_{scene_name}",
            command=command,
            stdout_log=paths["logs"] / f"inference_{scene_name}.stdout.log",
            stderr_log=paths["logs"] / f"inference_{scene_name}.stderr.log",
            completion_check=lambda s=scene_name: inference_complete(paths, s, args.timesteps),
            force=args.force,
            skip_existing=args.skip_existing,
        )
        inference_results.append({"scene_name": scene_name, **result})

        if result["status"] == "completed":
            print(f"completed in {result['duration_seconds']:.2f}s")
        elif result["status"] == "skipped":
            print("skipped")
        else:
            print("failed")
            if args.stop_on_error:
                break

    stage_results["inference"] = inference_results
    write_summary(args, paths, scene_names, stage_results)

    any_failures = any(
        result["status"] == "failed"
        for key, result in stage_results.items()
        if key != "inference"
    ) or any(result["status"] == "failed" for result in inference_results)

    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())