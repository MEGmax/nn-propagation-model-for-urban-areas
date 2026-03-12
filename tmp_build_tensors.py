from pathlib import Path
import argparse
import json
import numpy as np


def build_tensors(scenes_root: Path, out_input: Path, out_target: Path) -> int:
    out_input.mkdir(parents=True, exist_ok=True)
    out_target.mkdir(parents=True, exist_ok=True)

    count = 0
    for scene_dir in sorted([d for d in scenes_root.iterdir() if d.is_dir()]):
        elev_files = list(scene_dir.glob('elevation*.npy'))
        rss_files = list(scene_dir.glob('rss_values*.npy'))
        meta_file = scene_dir / 'tx_metadata.json'
        if not elev_files or not rss_files or not meta_file.exists():
            continue

        elevation = np.load(elev_files[0]).astype(np.float32)
        rss = np.load(rss_files[0]).astype(np.float32)
        if rss.ndim == 2:
            rss = rss[None, ...]

        h, w = int(rss.shape[1]), int(rss.shape[2])

        eh, ew = elevation.shape
        yi = np.clip((np.linspace(0, eh - 1, h)).round().astype(int), 0, eh - 1)
        xi = np.clip((np.linspace(0, ew - 1, w)).round().astype(int), 0, ew - 1)
        elevation_rs = elevation[np.ix_(yi, xi)].astype(np.float32)

        with open(meta_file, 'r') as file:
            tx_meta = json.load(file)

        tx_pos = np.array(tx_meta.get('tx_position', [0, 0, 0]), dtype=np.float32).reshape(-1)
        tx_x, tx_y = (float(tx_pos[0]), float(tx_pos[1])) if tx_pos.size >= 2 else (0.0, 0.0)

        freq_val = tx_meta.get('frequency', 2.4e9)
        if isinstance(freq_val, list):
            freq_hz = float(np.array(freq_val).reshape(-1)[0])
        else:
            freq_hz = float(freq_val)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xx_centered = xx - w / 2.0
        yy_centered = yy - h / 2.0
        distance_map = np.sqrt((xx_centered - tx_x) ** 2 + (yy_centered - tx_y) ** 2).astype(np.float32)

        wavelength = 3e8 / max(freq_hz, 1.0)
        distance_map = (distance_map / wavelength).astype(np.float32)
        freq_ghz = np.full((h, w), freq_hz / 1e9, dtype=np.float32)

        input_tensor = np.stack([elevation_rs, distance_map, freq_ghz], axis=-1).astype(np.float32)

        rss_safe = np.where(rss > 0, rss, 1e-12)
        rss_dbm = (10.0 * np.log10(rss_safe) + 30.0).astype(np.float32)
        target_tensor = np.transpose(rss_dbm, (1, 2, 0)).astype(np.float32)

        scene_name = scene_dir.name
        np.save(out_input / f'{scene_name}_input.npy', input_tensor)
        np.save(out_target / f'{scene_name}_target.npy', target_tensor)
        count += 1

    return count


def main():
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description='Build input/target tensors from generated scenes.')
    parser.add_argument(
        '--scenes-root',
        type=Path,
        default=project_root / 'scene_generation' / 'automated_scenes',
        help='Directory containing scene folders.'
    )
    parser.add_argument(
        '--output-input',
        type=Path,
        default=project_root / 'model_input' / 'data' / 'training' / 'input',
        help='Output directory for *_input.npy files.'
    )
    parser.add_argument(
        '--output-target',
        type=Path,
        default=project_root / 'model_input' / 'data' / 'training' / 'target',
        help='Output directory for *_target.npy files.'
    )
    args = parser.parse_args()

    if not args.scenes_root.exists():
        raise FileNotFoundError(f'Scenes root not found: {args.scenes_root}')

    count = build_tensors(args.scenes_root, args.output_input, args.output_target)
    print(f'created_pairs={count}')
    print(f'input_dir={args.output_input}')
    print(f'target_dir={args.output_target}')


if __name__ == '__main__':
    main()
