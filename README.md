# Person and Vehicle Detection in Thermal UAV Imagery

**Night-time and low-visibility detection of people and vehicles** in infrared images captured by UAV thermal cameras, and **analysis of the performance gap** between the visible-light and thermal domains.

<p align="center"><img src="docs/assets/pipeline.svg" width="100%" alt="Pipeline"></p>

## Problem

At night and in low-visibility conditions, UAVs rely on thermal cameras. However, most detection models are trained on visible-light images (e.g. COCO). Thermal images carry no color or texture information; objects appear through temperature differences. As a result, models trained on visible light suffer a large performance drop on thermal imagery.

## Approach

| Step | Method | File |
|---|---|---|
| 1. Data preparation | Validate the HIT-UAV folder structure, class distribution and training YAML | `src/prepare_data.py` |
| 2. Baseline | Evaluate a COCO-pretrained model directly (zero-shot) on thermal data | `src/compare_domains.py` |
| 3. Domain adaptation | Fine-tune the same model on HIT-UAV; color augmentation off, brightness augmentation on | `src/train.py` |
| 4. Comparison | Per-class precision / recall of both models on the same test images | `src/compare_domains.py` |

## Data

- **HIT-UAV** (Suo et al., *Scientific Data*, 2023): ~2,900 infrared thermal images captured from UAVs at various altitudes and angles. Classes: Person, Car, Bicycle, OtherVehicle, DontCare.
- Download: [Kaggle](https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset) or the [official repository](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset).

## Installation

```bash
git clone https://github.com/mahdildarr/thermal-detection-hituav.git
cd thermal-detection-hituav
pip install -r requirements.txt
```

The quickest way to run everything is the `notebooks/colab_thermal_detection.ipynb` notebook on Google Colab (GPU enabled).

## Usage

```bash
# 1) Data preparation
python src/prepare_data.py --root <downloaded folder> --out data/hituav.yaml

# 2) Thermal fine-tuning
python src/train.py --data data/hituav.yaml --model yolo11s.pt --epochs 60 --name yolo11s_thermal

# 3) Visible-light ↔ thermal comparison
python src/compare_domains.py --data data/hituav.yaml --thermal runs/thermal/yolo11s_thermal/weights/best.pt
```

## Evaluation metrics

- Thermal model: mAP@0.5, mAP@0.5:0.95, per-class AP
- Domain comparison: per-class precision / recall at the same confidence threshold (0.25), IoU ≥ 0.5

## Roadmap

- [x] Data preparation, training and comparison pipeline
- [ ] Zero-shot baseline
- [ ] Thermal fine-tuning
- [ ] Effect of augmentation strategies (brightness, blur, scale)
- [ ] Error analysis by altitude and viewing angle

## Project structure

```
├── data/                      # generated YAML files
├── notebooks/colab_thermal_detection.ipynb
├── src/
│   ├── prepare_data.py        # dataset checks and YAML
│   ├── train.py               # thermal fine-tuning
│   ├── compare_domains.py     # visible-light ↔ thermal comparison
│   └── utils.py               # IoU, label parsing, matching
└── requirements.txt
```

## References

- Suo et al., *HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection*, Scientific Data, 2023
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## License

MIT
