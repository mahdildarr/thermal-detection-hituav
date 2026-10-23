import numpy as np
import pandas as pd
import pytest

from thermaldet.boxes import Detections
from thermaldet.compare_domains import class_mapper, evaluate_model
from thermaldet.hituav import from_official_release, parse_metadata
from thermaldet.metrics import CocoCollector, recall_by

NAMES = ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]


def test_metadata_from_file_name():
    assert parse_metadata("x/1_130_50_0_03928.jpg") == {"scenario": "night", "altitude": 130, "angle": 50,
                                                        "weather": "sunny", "frame": 3928}
    assert parse_metadata("random.jpg") == {}


def test_coco_class_mapping():
    coco = {i: f"c{i}" for i in range(80)}
    coco.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    assert class_mapper(coco, NAMES) == {0: 0, 1: 2, 2: 1, 3: 3, 5: 3, 7: 3}
    # a model trained on HIT-UAV maps by name, and never predicts DontCare
    assert class_mapper(dict(enumerate(NAMES)), NAMES) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_dontcare_is_a_crowd_region():
    c = CocoCollector([0, 1, 2, 3], NAMES)
    gt = np.array([[10, 10, 40, 40]], np.float32)
    det = Detections(np.array([[10, 10, 40, 40], [100, 100, 130, 130]], np.float32), np.array([0, 0]),
                     np.array([0.5, 0.9]))
    c.add(0, 640, 512, gt, [0], det, ignore_boxes=[[95, 95, 135, 135]])
    r = c.evaluate()
    assert r["AP50"] == pytest.approx(1.0) and r["per_class_AP50"]["Person"] == pytest.approx(1.0)


def test_recall_by_condition_with_interval():
    rows = [{"image": f"i{k}", "altitude": 60 if k < 10 else 130, "matched": (k % 2 == 0) or k < 10}
            for k in range(20) for _ in range(3)]
    t = recall_by(pd.DataFrame(rows), "altitude", n_boot=200)
    assert t.set_index("altitude").loc[60, "recall"] == 1.0
    lo, hi = t.set_index("altitude").loc[130, ["ci_low", "ci_high"]]
    assert lo <= 0.5 <= hi


def test_official_layout_and_oracle_evaluation(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from thermaldet.boxes import label_path_for, load_yolo_labels

    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools" / "make_synthetic_hituav.py"), "--out", str(tmp_path / "raw")],
                   check=True)
    data = from_official_release(tmp_path / "raw", tmp_path / "yolo")
    images = sorted((data / "images" / "test").glob("*.jpg"))
    assert len(images) == 6 and len(list((data / "labels" / "train").glob("*.txt"))) == 4

    def oracle(img_path):  # returns the ground truth, DontCare included (must be ignored, not penalised)
        cls, boxes = load_yolo_labels(label_path_for(img_path), 640, 512)
        return Detections(boxes, cls, np.full(len(cls), 0.9, np.float32))

    mapping = class_mapper(dict(enumerate(NAMES)), NAMES)
    objects, preds, ap = evaluate_model(oracle, mapping, images, NAMES)
    assert objects["matched"].all() and preds["tp"].all()
    assert set(objects.columns) >= {"scenario", "altitude", "angle", "size"}
    assert ap["AP50"] == pytest.approx(1.0)
