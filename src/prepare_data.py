"""Locate the HIT-UAV folder structure, print dataset statistics and write a data YAML for training.

Expected layout (Kaggle release): <root>/images/{train,val,test} and <root>/labels/{train,val,test}
Class names are read from the YAML shipped with the dataset; a default list is used if none is found.

Example:
    python src/prepare_data.py --root /content/hit-uav --out data/hituav.yaml
"""

import argparse
import os
from collections import Counter
from glob import glob

import yaml

DEFAULT_NAMES = ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]
SPLITS = ["train", "val", "test"]


def find_dataset_root(root):
    """Return the first directory under root that contains images/train."""
    for d, subdirs, _ in os.walk(root):
        if os.path.basename(d) == "images" and "train" in subdirs:
            return os.path.dirname(d)
    raise SystemExit(f"No 'images/train' structure found under: {root}")


def find_names(root):
    for y in glob(os.path.join(root, "**", "*.yaml"), recursive=True):
        try:
            data = yaml.safe_load(open(y))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and "names" in data:
            names = data["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            print(f"Class names read from: {y}")
            return list(names)
    print("No YAML found in the dataset, using default class names.")
    return DEFAULT_NAMES


def split_stats(root, split, names):
    imgs = glob(os.path.join(root, "images", split, "*"))
    counts = Counter()
    for lbl in glob(os.path.join(root, "labels", split, "*.txt")):
        for line in open(lbl):
            parts = line.split()
            if parts:
                counts[int(float(parts[0]))] += 1
    return len(imgs), {names[k] if k < len(names) else k: v for k, v in sorted(counts.items())}


def main():
    p = argparse.ArgumentParser(description="HIT-UAV preparation")
    p.add_argument("--root", required=True, help="Folder of the downloaded dataset")
    p.add_argument("--out", default="data/hituav.yaml")
    a = p.parse_args()

    root = os.path.abspath(find_dataset_root(a.root))
    names = find_names(a.root)
    print(f"Dataset root: {root}\nClasses: {names}\n")

    cfg = {"path": root, "names": {i: n for i, n in enumerate(names)}}
    for split in SPLITS:
        if os.path.isdir(os.path.join(root, "images", split)):
            cfg[split] = f"images/{split}"
            n_img, counts = split_stats(root, split, names)
            print(f"{split:<5}: {n_img} images | object counts: {counts}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"\nYAML written: {a.out}")


if __name__ == "__main__":
    main()
