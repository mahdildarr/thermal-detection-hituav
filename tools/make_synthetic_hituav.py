"""Create a tiny synthetic dataset in the official HIT-UAV release layout (file names carry the
capture metadata) to exercise the pipeline without real data.

    python tools/make_synthetic_hituav.py --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SPLITS = {"train": 4, "val": 2, "test": 6}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    jpg, sets = root / "normal_xml" / "JPEGImages", root / "normal_xml" / "ImageSets" / "Main"
    jpg.mkdir(parents=True, exist_ok=True)
    sets.mkdir(parents=True, exist_ok=True)
    frame = 0
    for split, n in SPLITS.items():
        (root / "yolo_labels" / split).mkdir(parents=True, exist_ok=True)
        stems = []
        for _ in range(n):
            frame += 1
            scenario, alt, ang = int(rng.integers(0, 2)), int(rng.choice([60, 90, 130])), int(rng.choice([30, 60, 90]))
            stem = f"{scenario}_{alt}_{ang}_0_{frame:05d}"
            img = np.full((512, 640, 3), 40, np.uint8)
            lines = []
            for _ in range(6):
                c = int(rng.integers(0, 5))
                s = int(rng.integers(10, 60))
                x, y = int(rng.integers(0, 640 - s)), int(rng.integers(0, 512 - s))
                cv2.rectangle(img, (x, y), (x + s, y + s), (230, 230, 230), -1)  # warm object
                lines.append(f"{c} {(x + s / 2) / 640:.6f} {(y + s / 2) / 512:.6f} {s / 640:.6f} {s / 512:.6f}")
            cv2.imwrite(str(jpg / f"{stem}.jpg"), img)
            (root / "yolo_labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            stems.append(stem)
        (sets / f"{split}.txt").write_text("\n".join(stems))
    print(f"Synthetic HIT-UAV-style data -> {root}")


if __name__ == "__main__":
    main()
