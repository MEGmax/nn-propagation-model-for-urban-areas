"""
split_scenes.py
 
Splits an `automated_scenes` directory of scene folders into
train / val / test sets (70 / 15 / 15) by random shuffle,
then copies each scene into the corresponding output folder.
 
Usage:
    python split_scenes.py \
        --input_dir  /Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes \
        --output_dir path/to/split_dataset \
        --seed       42          # optional, for reproducibility
"""
 
import argparse
import random
import shutil
from pathlib import Path
 
 
# ── defaults ──────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15          # remainder goes here
SPLITS      = ("train", "val", "test")
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split automated_scenes into train/val/test.")
    p.add_argument("--input_dir",  type=Path, default=Path("automated_scenes"),
                   help="Root folder that contains scene0, scene1, … sub-folders.")
    p.add_argument("--output_dir", type=Path, default=Path("split_dataset"),
                   help="Destination root.  Will be created if it does not exist.")
    p.add_argument("--seed",       type=int,  default=42,
                   help="Random seed for reproducibility.")
    return p.parse_args()
 
 
def discover_scenes(input_dir: Path) -> list[Path]:
    """Return sorted list of immediate sub-directories that look like scenes."""
    scenes = sorted(
        p for p in input_dir.iterdir()
        if p.is_dir()
    )
    if not scenes:
        raise FileNotFoundError(f"No scene sub-folders found in '{input_dir}'.")
    return scenes
 
 
def compute_splits(scenes: list[Path], seed: int) -> dict[str, list[Path]]:
    """Randomly shuffle scenes then slice into train / val / test."""
    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
 
    n       = len(shuffled)
    n_train = round(n * TRAIN_RATIO)
    n_val   = round(n * VAL_RATIO)
    # test gets everything that remains (avoids rounding gaps)
 
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }
 
 
def copy_splits(splits: dict[str, list[Path]], output_dir: Path) -> None:
    """Copy each scene folder into output_dir/<split>/<scene_name>."""
    for split, scene_paths in splits.items():
        split_root = output_dir / split
        split_root.mkdir(parents=True, exist_ok=True)
 
        for scene_path in scene_paths:
            dest = split_root / scene_path.name
            if dest.exists():
                print(f"  [skip]  {dest} already exists – remove it to re-copy.")
                continue
            print(f"  [{split:5s}]  {scene_path.name}  →  {dest}")
            shutil.copytree(scene_path, dest)
 
 
def print_summary(splits: dict[str, list[Path]]) -> None:
    total = sum(len(v) for v in splits.values())
    print("\n── Split summary ────────────────────────────────")
    for split, scenes in splits.items():
        pct = 100 * len(scenes) / total if total else 0
        names = ", ".join(s.name for s in scenes)
        print(f"  {split:5s}  {len(scenes):3d} scenes ({pct:.0f}%)  [{names}]")
    print(f"  {'total':5s}  {total:3d} scenes")
    print("─────────────────────────────────────────────────\n")
 
 
def main() -> None:
    args = parse_args()
 
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: '{args.input_dir}'")
 
    print(f"\nScanning '{args.input_dir}' for scenes …")
    scenes = discover_scenes(args.input_dir)
    print(f"Found {len(scenes)} scenes: {[s.name for s in scenes]}")
 
    splits = compute_splits(scenes, seed=args.seed)
    print_summary(splits)
 
    print(f"Copying into '{args.output_dir}' …\n")
    copy_splits(splits, args.output_dir)
 
    print("\nDone! Output structure:")
    for split in SPLITS:
        split_dir = args.output_dir / split
        count = len(list(split_dir.iterdir())) if split_dir.exists() else 0
        print(f"  {args.output_dir}/{split}/   ({count} scenes)")
 
 
if __name__ == "__main__":
    main()
 