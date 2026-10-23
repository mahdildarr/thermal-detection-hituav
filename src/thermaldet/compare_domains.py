"""Quantify the visible-light -> thermal domain gap and analyse errors by capture condition.

Models are evaluated on the same thermal images, e.g.

1. a visible-light model: COCO-pretrained YOLO that has never seen thermal imagery, with COCO classes
   mapped to HIT-UAV (person -> Person, car -> Car, bicycle -> Bicycle, motorcycle / bus / truck ->
   OtherVehicle);
2. a thermal model fine-tuned on HIT-UAV.

Reported for each model:

* precision / recall per class at a fixed confidence threshold (IoU >= 0.5),
* COCO AP, AP50 and per-class AP50 (pycocotools),
* recall by time of day, flight altitude, camera angle and object size, with 95% bootstrap
  intervals over images (from the capture metadata in the HIT-UAV file names).

DontCare boxes are treated as ignored regions: detections on them are not false positives.

Example::

    python -m thermaldet.compare_domains --data data/hituav.yaml \
        --model "Visible (COCO)=yolo11s.pt" --model "Thermal (HIT-UAV)=runs/thermal/yolo11s_thermal/weights/best.pt"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .boxes import (
    SIZE_BINS,
    Detections,
    box_iou,
    label_path_for,
    load_yolo_labels,
    match_detections,
    outside_ignored,
    size_bin,
)
from .hituav import parse_metadata
from .metrics import CocoCollector, recall_by

COCO_TO_HIT = {"person": "person", "car": "car", "bicycle": "bicycle",
               "motorcycle": "othervehicle", "bus": "othervehicle", "truck": "othervehicle"}
CONDITIONS = ["scenario", "altitude", "angle", "size"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Visible-light vs thermal model comparison")
    p.add_argument("--data", default="data/hituav.yaml")
    p.add_argument("--model", action="append", required=True, metavar="LABEL=WEIGHTS",
                   help="Model to evaluate; repeat for several. COCO class names are mapped automatically.")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="Operating-point confidence threshold")
    p.add_argument("--ap-conf", type=float, default=0.001, help="Inference threshold used for AP")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/thermal/domain_comparison")
    return p.parse_args(argv)


def load_cfg(path, split):
    cfg = yaml.safe_load(Path(path).read_text())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    sp = cfg.get(split) or cfg.get("val")
    img_dir = Path(sp) if Path(sp).is_absolute() else Path(cfg.get("path", "")) / sp
    return names, img_dir


def class_mapper(model_names: dict, data_names: list[str]) -> dict[int, int]:
    """Model class index -> dataset class index. Uses the COCO mapping when the model has COCO classes."""
    target = {n.lower(): i for i, n in enumerate(data_names)}
    is_coco = "person" in {n.lower() for n in model_names.values()} and len(model_names) == 80
    mapping = {}
    for idx, n in model_names.items():
        key = COCO_TO_HIT.get(n.lower()) if is_coco else n.lower()
        if key in target and key != "dontcare":
            mapping[int(idx)] = target[key]
    return mapping


class YoloDetector:
    def __init__(self, weights, imgsz, conf, device=None):
        from ultralytics import YOLO

        self.model, self.imgsz, self.conf, self.device = YOLO(weights), imgsz, conf, device
        self.names = self.model.names

    def __call__(self, img_path) -> Detections:
        r = self.model.predict(str(img_path), imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        b = r[0].boxes
        return Detections(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())


def evaluate_model(detect, mapping, images, names, conf=0.25, iou=0.5):
    """Returns (objects, predictions, coco_metrics). One row per ground-truth object / prediction."""
    ignore_id = {i for i, n in enumerate(names) if n.lower() == "dontcare"}
    eval_ids = [i for i in range(len(names)) if i not in ignore_id]
    coco = CocoCollector(eval_ids, names)
    objects, preds = [], []
    for k, img_path in enumerate(images):
        h, w = cv2.imread(str(img_path)).shape[:2]
        meta = parse_metadata(img_path)
        cls, boxes = load_yolo_labels(label_path_for(img_path), w, h)
        dc = np.isin(cls, list(ignore_id))
        gt_cls, gt_boxes, ignore = cls[~dc], boxes[~dc], boxes[dc]

        det = detect(img_path)
        keep = np.array([c in mapping for c in det.cls], dtype=bool)
        det = Detections(det.boxes[keep], np.array([mapping[c] for c in det.cls[keep]], dtype=int), det.conf[keep])
        coco.add(k, w, h, gt_boxes, gt_cls, det, ignore)

        could_match = box_iou(det.boxes, gt_boxes).max(axis=1, initial=0) >= iou
        op = det.subset((det.conf >= conf) & (could_match | outside_ignored(det.boxes, ignore)))
        matched, tp = match_detections(gt_boxes, gt_cls, op.boxes, op.cls, op.conf, iou)
        sizes = size_bin(gt_boxes)
        for c, m, s in zip(gt_cls, matched, sizes):
            objects.append({"image": img_path.name, "cls": names[c], "matched": bool(m), "size": SIZE_BINS[s][0],
                            **meta})
        for c, t, s in zip(op.cls, tp, op.conf):
            preds.append({"image": img_path.name, "cls": names[c], "tp": bool(t), "conf": float(s), **meta})
    cols = ["image", "cls", "tp", "conf"]
    return pd.DataFrame(objects), pd.DataFrame(preds, columns=cols if not preds else None), coco.evaluate()


def pr_table(objects: pd.DataFrame, preds: pd.DataFrame, classes: list[str]) -> dict:
    out = {}
    for c in [*classes, "Overall"]:
        o = objects if c == "Overall" else objects[objects["cls"] == c]
        p = preds if c == "Overall" else preds[preds["cls"] == c]
        out[c] = {"precision": p["tp"].mean() if len(p) else float("nan"),
                  "recall": o["matched"].mean() if len(o) else float("nan"), "objects": len(o)}
    return out


def plot_conditions(tables: dict, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (cond, xlabel) in zip(axes, [("altitude", "Flight altitude (m)"), ("angle", "Camera angle (°)"),
                                         ("scenario", "Time of day")]):
        for label, t in tables.items():
            d = t[cond]
            if d.empty:
                continue
            x = np.arange(len(d)) if cond == "scenario" else d[cond].to_numpy()
            ax.plot(x, d["recall"], marker="o", label=label)
            ax.fill_between(x, d["ci_low"], d["ci_high"], alpha=0.2)
            if cond == "scenario":
                ax.set_xticks(x, d[cond])
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall (95% CI)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(v):
    return "-" if v != v else f"{v:.3f}"  # NaN check


def write_report(results: dict, classes: list[str], args, out: Path) -> str:
    lines = ["## Precision / recall per class", "",
             f"Confidence {args.conf}, IoU {args.iou}. DontCare regions are ignored.", "",
             "| Model | " + " | ".join(f"{c} P / R" for c in [*classes, "Overall"]) + " |",
             "|---|" + "---|" * (len(classes) + 1)]
    for label, r in results.items():
        lines.append(f"| {label} | " + " | ".join(f"{fmt(v['precision'])} / {fmt(v['recall'])}"
                                                   for v in r["pr"].values()) + " |")
    lines += ["", "## COCO AP", "", "| Model | AP | AP50 | AP_small | AP_medium | "
              + " | ".join(f"AP50 {c}" for c in classes) + " |", "|---|---|---|---|---|" + "---|" * len(classes)]
    for label, r in results.items():
        a = r["ap"]
        lines.append(f"| {label} | {fmt(a['AP'])} | {fmt(a['AP50'])} | {fmt(a['AP_small'])} | {fmt(a['AP_medium'])} | "
                     + " | ".join(fmt(a["per_class_AP50"].get(c, float("nan"))) for c in classes) + " |")
    for cond in CONDITIONS:
        lines += ["", f"## Recall by {cond}", "", "| Model | " + cond + " | Images | Objects | Recall | 95% CI |",
                  "|---|---|---|---|---|---|"]
        for label, r in results.items():
            for row in r["conditions"][cond].to_dict("records"):
                v = row[cond]
                v = int(v) if isinstance(v, float) and v.is_integer() else v
                lines.append(f"| {label} | {v} | {int(row['images'])} | {int(row['objects'])} | {row['recall']:.3f} | "
                             f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def run(args, detectors: dict | None = None) -> dict:
    """``detectors``: label -> (callable, model class names), injectable for tests."""
    names, img_dir = load_cfg(args.data, args.split)
    images = sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.png")])
    if args.limit:
        images = images[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if detectors is None:
        detectors = {}
        for spec in args.model:
            label, weights = spec.split("=", 1)
            d = YoloDetector(weights, args.imgsz, args.ap_conf, args.device)
            detectors[label] = (d, d.names)
    classes = [n for n in names if n.lower() != "dontcare"]
    print(f"{len(images)} thermal images from {img_dir}")

    results = {}
    for label, (detect, model_names) in detectors.items():
        mapping = class_mapper(model_names, names)
        objects, preds, ap = evaluate_model(detect, mapping, images, names, args.conf, args.iou)
        slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()
        objects.to_csv(out / f"objects_{slug}.csv", index=False)
        preds.to_csv(out / f"predictions_{slug}.csv", index=False)
        conds = {c: recall_by(objects, c, args.bootstrap) if c in objects else pd.DataFrame() for c in CONDITIONS}
        results[label] = {"pr": pr_table(objects, preds, classes), "ap": ap, "conditions": conds}
        print(f"done: {label}  AP50={ap['AP50']:.3f}")

    plot_conditions({k: v["conditions"] for k, v in results.items()}, out / "recall_by_condition.png")
    text = write_report(results, classes, args, out)
    serial = {k: {"pr": v["pr"], "ap": v["ap"],
                  "conditions": {c: d.to_dict("records") for c, d in v["conditions"].items()}}
              for k, v in results.items()}
    (out / "metrics.json").write_text(json.dumps(serial, indent=2, default=float))
    print("\n" + text + f"\nSaved to {out}/")
    return results


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
