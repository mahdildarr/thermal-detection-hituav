"""Train a YOLO detector on HIT-UAV thermal imagery.

Example:
    python src/train.py --data data/hituav.yaml --model yolo11s.pt --epochs 60 --imgsz 640 --name yolo11s_thermal
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Thermal detection training")
    p.add_argument("--data", default="data/hituav.yaml")
    p.add_argument("--model", default="yolo11s.pt", help="COCO-pretrained (visible-light) initial weights")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="yolo11s_thermal")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project="runs/thermal",
        name=a.name,
        exist_ok=True,
        seed=a.seed,
        patience=20,
        # Thermal images carry single-channel intensity; hue/saturation augmentation is meaningless
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        plots=True,
    )
    split = "test"
    try:
        metrics = model.val(data=a.data, imgsz=a.imgsz, split=split, project="runs/thermal", name=f"{a.name}_{split}", exist_ok=True)
    except Exception:  # noqa: BLE001  fall back to the validation split if there is no test split
        split = "val"
        metrics = model.val(data=a.data, imgsz=a.imgsz, split=split, project="runs/thermal", name=f"{a.name}_{split}", exist_ok=True)
    print(f"\n=== {split} results ===")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    for k, c in enumerate(metrics.box.ap_class_index):
        print(f"  {metrics.names[int(c)]:<14} AP@0.5 = {metrics.box.ap50[k]:.4f}")
    print(f"Weights      : runs/thermal/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
