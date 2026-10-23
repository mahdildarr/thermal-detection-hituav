"""COCO AP with DontCare as crowd regions, and recall by capture condition with bootstrap intervals."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from .boxes import Detections


class CocoCollector:
    """Accumulates ground truth and detections and runs the official COCO evaluation.

    DontCare boxes are added as ``iscrowd=1``: COCOeval then neither rewards nor penalises
    detections on them.
    """

    def __init__(self, class_ids: list[int], names: list[str]):
        self.cats = [{"id": int(i), "name": names[i]} for i in class_ids]
        self.images, self.gt, self.dt = [], [], []

    def add(self, image_id: int, width: int, height: int, gt_boxes, gt_cls, det: Detections, ignore_boxes=()):
        self.images.append({"id": image_id, "width": width, "height": height})

        def xywh(b):
            return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]

        for b, c in zip(gt_boxes, gt_cls):
            w = xywh(b)
            self.gt.append({"id": len(self.gt) + 1, "image_id": image_id, "category_id": int(c), "bbox": w,
                            "area": w[2] * w[3], "iscrowd": 0})
        for b in ignore_boxes:
            w = xywh(b)
            for cat in self.cats:
                self.gt.append({"id": len(self.gt) + 1, "image_id": image_id, "category_id": cat["id"], "bbox": w,
                                "area": w[2] * w[3], "iscrowd": 1})
        for b, c, s in zip(det.boxes, det.cls, det.conf):
            self.dt.append({"image_id": image_id, "category_id": int(c), "bbox": xywh(b), "score": float(s)})

    def evaluate(self) -> dict:
        """Overall AP / AP50 / AP_small... and per-class AP50."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        nan = float("nan")
        if not self.dt:
            return {"AP": 0.0, "AP50": 0.0, "per_class_AP50": {c["name"]: 0.0 for c in self.cats}}
        gt = COCO()
        gt.dataset = {"images": self.images, "annotations": self.gt, "categories": self.cats}
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            ev = COCOeval(gt, gt.loadRes(self.dt), "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = [float(v) if v >= 0 else nan for v in ev.stats]
        # precision: [T, R, K, A, M]; IoU 0.5 is index 0, area "all" index 0, maxDets 100 index -1
        prec = ev.eval["precision"][0, :, :, 0, -1]
        per_class = {}
        for k, cat in enumerate(self.cats):
            p = prec[:, k]
            per_class[cat["name"]] = float(p[p > -1].mean()) if (p > -1).any() else nan
        return {"AP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
                "per_class_AP50": per_class}


def recall_by(objects: pd.DataFrame, by: str, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Recall per value of ``by`` with a 95% bootstrap interval that resamples *images*.

    ``objects`` has one row per ground-truth object with columns image, matched and ``by``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for value, g in objects.groupby(by):
        per_img = g.groupby("image")["matched"].agg(["sum", "count"]).to_numpy()
        rec = per_img[:, 0].sum() / per_img[:, 1].sum()
        lo = hi = float("nan")
        if n_boot and len(per_img) > 1:
            idx = rng.integers(0, len(per_img), (n_boot, len(per_img)))
            boot = per_img[idx, 0].sum(1) / per_img[idx, 1].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({by: value, "images": len(per_img), "objects": int(per_img[:, 1].sum()), "recall": rec,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
