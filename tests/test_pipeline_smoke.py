"""End-to-end: synthetic official release -> prepare_data -> comparison report (injected detectors)."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from thermaldet.boxes import Detections, label_path_for, load_yolo_labels
from thermaldet.compare_domains import parse_args, run

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]


def test_end_to_end(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_hituav.py"), "--out", str(tmp_path / "raw")],
                   check=True)
    subprocess.run([sys.executable, "-m", "thermaldet.prepare_data", "--root", str(tmp_path / "raw"),
                    "--work-dir", str(tmp_path / "hituav"), "--out", str(tmp_path / "hituav.yaml")],
                   check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})

    def oracle(drop_small: bool):
        def detect(img_path):
            cls, boxes = load_yolo_labels(label_path_for(img_path), 640, 512)
            keep = (boxes[:, 2] - boxes[:, 0] >= 32) if drop_small else np.ones(len(cls), bool)
            return Detections(boxes[keep], cls[keep], np.full(int(keep.sum()), 0.9, np.float32))
        return detect

    model_names = dict(enumerate(NAMES))
    args = parse_args(["--data", str(tmp_path / "hituav.yaml"), "--model", "unused=x", "--bootstrap", "50",
                       "--out", str(tmp_path / "out")])
    res = run(args, detectors={"Weak": (oracle(True), model_names), "Oracle": (oracle(False), model_names)})
    assert res["Oracle"]["pr"]["Overall"]["recall"] == 1.0
    assert res["Weak"]["pr"]["Overall"]["recall"] < 1.0
    small = res["Weak"]["conditions"]["size"].set_index("size")
    assert small.loc["small", "recall"] == 0.0
    for f in ("report.md", "metrics.json", "recall_by_condition.png", "objects_oracle.csv"):
        assert (tmp_path / "out" / f).exists()
