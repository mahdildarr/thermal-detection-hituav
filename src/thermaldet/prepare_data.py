"""Locate HIT-UAV, print per-split statistics (classes, day / night, altitude, angle) and write a data YAML.

Accepts the Kaggle release (images/ + labels/) or the official GitHub release (normal_xml/ +
yolo_labels/), which is converted with symlinks.

Example::

    python -m thermaldet.prepare_data --root /content/hit-uav --out data/hituav.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from .hituav import CLASSES, SPLITS, data_yaml, parse_metadata, resolve_root


def split_table(root: Path, split: str) -> pd.DataFrame:
    rows = []
    for img in sorted((root / "images" / split).glob("*.jpg")):
        lab = root / "labels" / split / f"{img.stem}.txt"
        counts = Counter(int(float(line.split()[0])) for line in lab.read_text().splitlines()
                         if line.strip()) if lab.exists() else Counter()
        rows.append({"image": img.name, **parse_metadata(img), **{CLASSES[k]: v for k, v in counts.items()}})
    return pd.DataFrame(rows).fillna({c: 0 for c in CLASSES})


def main(argv=None):
    p = argparse.ArgumentParser(description="HIT-UAV preparation")
    p.add_argument("--root", required=True, help="Folder of the downloaded dataset")
    p.add_argument("--work-dir", default="data/hituav", help="Where the official release is re-arranged")
    p.add_argument("--out", default="data/hituav.yaml")
    a = p.parse_args(argv)

    root = resolve_root(a.root, a.work_dir)
    cfg = data_yaml(root)
    print(f"Dataset root: {root}")
    for split in SPLITS:
        if split not in cfg:
            continue
        df = split_table(root, split)
        objs = {c: int(df[c].sum()) for c in CLASSES if c in df}
        meta = ""
        if "scenario" in df:
            sc = {k: int(v) for k, v in df["scenario"].value_counts().items()}
            meta = (f" | {sc} | altitude {df['altitude'].min()}-{df['altitude'].max()} m"
                    f" | angle {df['angle'].min()}-{df['angle'].max()}°")
        print(f"{split:<5}: {len(df)} images | objects {objs}{meta}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"\nYAML written: {a.out}")


if __name__ == "__main__":
    main()
