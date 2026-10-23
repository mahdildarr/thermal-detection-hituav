"""HIT-UAV dataset: layout discovery, YOLO data YAML and per-image capture metadata.

Every HIT-UAV file name encodes how the image was captured::

    {scenario}_{altitude}_{angle}_{weather}_{frame}.jpg      e.g. 1_130_50_0_03928.jpg

scenario 0 = day, 1 = night; altitude 60-130 m; camera angle 30-90 degrees (90 = looking straight
down); weather 0 = sunny, 1 = rainy (Suo et al., 2023). The metadata is what allows the error analysis
by altitude, viewing angle and time of day.

Two layouts are supported:

* Kaggle / YOLO release: ``<root>/images/{train,val,test}`` and ``<root>/labels/{train,val,test}``
* official GitHub release: ``normal_xml/JPEGImages``, ``yolo_labels/{split}`` and the split lists in
  ``normal_xml/ImageSets/Main/{split}.txt``. It is converted into the first layout with symlinks.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

CLASSES = ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]
SPLITS = ("train", "val", "test")
SCENARIOS = {0: "day", 1: "night"}
WEATHER = {0: "sunny", 1: "rainy"}
_NAME = re.compile(r"^(\d)_(\d+)_(\d+)_(\d)_(\d+)$")


def parse_metadata(path) -> dict:
    """Capture conditions from a HIT-UAV file name (empty dict if the name does not follow the scheme)."""
    m = _NAME.match(Path(path).stem)
    if not m:
        return {}
    sc, alt, ang, we, frame = map(int, m.groups())
    return {"scenario": SCENARIOS.get(sc, str(sc)), "altitude": alt, "angle": ang,
            "weather": WEATHER.get(we, str(we)), "frame": frame}


def find_yolo_root(root) -> Path | None:
    """First directory under ``root`` that contains ``images/train``."""
    for d, subdirs, _ in os.walk(root):
        if os.path.basename(d) == "images" and "train" in subdirs:
            return Path(d).parent
    return None


def from_official_release(root, out) -> Path:
    """Build an images/labels split layout (symlinks) from the official GitHub release."""
    root, out = Path(root), Path(out)
    jpg, sets = root / "normal_xml" / "JPEGImages", root / "normal_xml" / "ImageSets" / "Main"
    labels = root / "yolo_labels"
    if not (jpg.is_dir() and sets.is_dir() and labels.is_dir()):
        raise SystemExit(f"Neither an images/train layout nor the official HIT-UAV release found under {root}")
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in (sets / f"{split}.txt").read_text().split():
            for src, dst in ((jpg / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg"),
                             (labels / split / f"{stem}.txt", out / "labels" / split / f"{stem}.txt")):
                if src.exists() and not dst.exists():
                    dst.symlink_to(src.resolve())
    return out


def resolve_root(root, work_dir="data/hituav") -> Path:
    found = find_yolo_root(root)
    return found if found is not None else from_official_release(root, work_dir)


def data_yaml(root: Path, names=CLASSES) -> dict:
    cfg = {"path": str(Path(root).resolve()), "names": dict(enumerate(names))}
    for split in SPLITS:
        if (Path(root) / "images" / split).is_dir():
            cfg[split] = f"images/{split}"
    return cfg
