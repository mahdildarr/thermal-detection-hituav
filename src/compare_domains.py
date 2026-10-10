"""Quantify the visible-light ↔ thermal domain gap.

Two models are compared on the same thermal test images:
  1. Visible-light model: COCO-pretrained YOLO that has never seen thermal imagery. COCO classes are mapped
     to HIT-UAV classes: person→Person, car→Car, bicycle→Bicycle, motorcycle/bus/truck→OtherVehicle.
  2. Thermal model: fine-tuned on HIT-UAV.

Per-class precision / recall (IoU ≥ 0.5) is reported for both. "DontCare" labels are excluded.

Example:
    python src/compare_domains.py --data data/hituav.yaml --thermal runs/thermal/yolo11s_thermal/weights/best.pt
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from utils import label_path_for, load_yolo_labels, match_detections

COCO_TO_HIT = {"person": "person", "car": "car", "bicycle": "bicycle",
               "motorcycle": "othervehicle", "bus": "othervehicle", "truck": "othervehicle"}


def parse_args():
    p = argparse.ArgumentParser(description="Visible-light vs. thermal model comparison")
    p.add_argument("--data", default="data/hituav.yaml")
    p.add_argument("--visible", default="yolo11s.pt", help="COCO-pretrained model")
    p.add_argument("--thermal", required=True, help="Model trained on thermal data")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="runs/thermal/domain_comparison.md")
    return p.parse_args()


def load_cfg(path, split):
    cfg = yaml.safe_load(open(path))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    sp = cfg.get(split) or cfg.get("val")
    img_dir = sp if os.path.isabs(sp) else os.path.join(cfg.get("path", ""), sp)
    return names, img_dir


def class_mapper(model_names, data_names, use_coco_map):
    """Build a dict mapping model class indices to dataset class indices."""
    target = {n.lower(): i for i, n in enumerate(data_names)}
    mapping = {}
    for idx, n in model_names.items():
        key = COCO_TO_HIT.get(n.lower()) if use_coco_map else n.lower()
        if key in target:
            mapping[int(idx)] = target[key]
    return mapping


def evaluate(model, images, data_names, mapping, ignore_ids, imgsz, conf):
    n_cls = len(data_names)
    tp, fp, gt_n = np.zeros(n_cls, int), np.zeros(n_cls, int), np.zeros(n_cls, int)
    for img_path in images:
        h, w = cv2.imread(img_path).shape[:2]
        gt_cls, gt_boxes = load_yolo_labels(label_path_for(img_path), w, h)
        keep = ~np.isin(gt_cls, list(ignore_ids))
        gt_cls, gt_boxes = gt_cls[keep], gt_boxes[keep]

        r = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sel = np.array([c in mapping for c in cls], dtype=bool)
        boxes = r.boxes.xyxy.cpu().numpy()[sel]
        scores = r.boxes.conf.cpu().numpy()[sel]
        pcls = np.array([mapping[c] for c in cls[sel]], dtype=int)

        gt_matched, pred_tp = match_detections(gt_boxes, gt_cls, boxes, pcls, scores)
        for c in range(n_cls):
            gt_n[c] += int((gt_cls == c).sum())
            tp[c] += int(((pcls == c) & pred_tp).sum())
            fp[c] += int(((pcls == c) & ~pred_tp).sum())
    return tp, fp, gt_n


def main():
    a = parse_args()
    names, img_dir = load_cfg(a.data, a.split)
    images = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
    if a.limit:
        images = images[: a.limit]
    ignore = {i for i, n in enumerate(names) if n.lower() == "dontcare"}
    print(f"{len(images)} thermal images: {img_dir}")

    rows = {}
    for label, weights, coco_map in [("Visible-light model (COCO)", a.visible, True), ("Thermal model (HIT-UAV)", a.thermal, False)]:
        model = YOLO(weights)
        mapping = class_mapper(model.names, names, coco_map)
        rows[label] = evaluate(model, images, names, mapping, ignore, a.imgsz, a.conf)
        print(f"Done: {label}")

    eval_ids = [i for i in range(len(names)) if i not in ignore]
    head = "| Model | " + " | ".join(f"{names[i]} P / R" for i in eval_ids) + " | Overall P / R |"
    lines = [head, "|---|" + "---|" * (len(eval_ids) + 1)]
    for label, (tp, fp, gt_n) in rows.items():
        cells = [f"{tp[i] / max(tp[i] + fp[i], 1):.2f} / {tp[i] / max(gt_n[i], 1):.2f}" for i in eval_ids]
        T, F, G = tp[eval_ids].sum(), fp[eval_ids].sum(), gt_n[eval_ids].sum()
        cells.append(f"{T / max(T + F, 1):.2f} / {T / max(G, 1):.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    gt_counts = next(iter(rows.values()))[2]
    lines += ["", "Ground-truth object counts: " + ", ".join(f"{names[i]}: {gt_counts[i]}" for i in eval_ids)]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    print("\n" + text + f"\n\nReport: {a.out}")


if __name__ == "__main__":
    main()
