# YOLO26-RD (CE-SPD): Contrast-Enhanced, Edge-Guided Road Damage Detection

A novel end-to-end (NMS-free) object detector for pavement distress — **alligator cracking, linear cracks, and patching** — built on the YOLO26 architecture and specialized for thin, low-contrast crack structures in grayscale road-survey imagery.

## Highlights

- **NMS-free end-to-end detection** — inherits YOLO26's `end2end` head (`reg_max: 1`), no NMS post-processing at deployment.
- **LearnableContrast stem** *(new module)* — a differentiable, end-to-end learnable CLAHE analogue: per-tile (8×8 grid), per-channel gamma/gain predicted from local image statistics with cross-tile context, bilinearly blended across tile boundaries. Unlike global image-adaptive filters (IA-YOLO, GDIP, ERUP-YOLO), contrast is adapted *locally*, matching why CLAHE outperforms global gamma on pavement imagery. Zero-initialized to identity, 494 parameters.
- **EdgeSPD** *(new module)* — Edge-guided Space-to-Depth downsampling. A fixed Sobel gradient prior gates features toward crack edges, then a lossless space-to-depth rearrangement replaces stride-2 convolution, so hairline cracks survive downsampling (+2 parameters per instance over plain SPD-Conv).
- **A2C2f area attention** at backbone P4 and head P3 — long-range context aggregation along elongated crack structures.
- **P2/4 high-resolution detection head** — 4 detection scales (P2–P5) for hairline cracks and small potholes.
- **BiFPN-style cross-scale skips** — backbone P3/P4 features concatenated directly into the bottom-up fusion path.

## Architecture

```
Input (B,3,H,W) in [0,1]
  └─ LearnableContrast          tile-wise adaptive gamma/gain, learnable CLAHE analogue
  └─ Conv P1/2 → Conv P2/4 → C3k2
  └─ EdgeSPD  P3/8              Sobel-gated space-to-depth (lossless ↓2)
  └─ C3k2
  └─ EdgeSPD  P4/16             Sobel-gated space-to-depth (lossless ↓2)
  └─ A2C2f (area attention)     elongated-crack context
  └─ Conv P5/32 → C3k2 → SPPF → C2PSA
Head (PANet + BiFPN-style skips)
  └─ top-down:  P5→P4 (C3k2) → P4→P3 (A2C2f) → P3→P2 (C3k2)
  └─ bottom-up: P2→P3 (+bb P3 skip) → P3→P4 (+bb P4 skip) → P4→P5
  └─ Detect(P2, P3, P4, P5)     end-to-end, NMS-free
```

## Model versions

| Version | Config | Description |
|---|---|---|
| v3–v6 | (training server) | Prior baselines; best v6: 0.771 mAP50 / 0.467 mAP50-95 |
| **v7** | [`models/yolo26-rd-v7.yaml`](models/yolo26-rd-v7.yaml) | Known-component recombination: Focus(SPD) + A2C2f + P2 head + BiFPN skips — ablation baseline |
| **v8** | [`models/yolo26-rd-v8.yaml`](models/yolo26-rd-v8.yaml) | **Full CE-SPD model**: v7 + LearnableContrast stem + EdgeSPD (novel modules) |

Baseline validation results (v3–v6, all classes):

| | v3 | v4 | v5 | v6 |
|---|---|---|---|---|
| mAP50 | 0.746 | 0.717 | 0.750 | **0.771** |
| mAP50-95 | 0.437 | 0.440 | 0.448 | **0.467** |
| Precision | 0.769 | 0.801 | 0.757 | **0.806** |
| Recall | 0.677 | 0.671 | 0.677 | **0.713** |

Per-class (v6): alligator 0.843 · crack 0.719 · patching 0.751 — **crack is the bottleneck class** that CE-SPD targets.

## Installation

This repository is a fork of Ultralytics with the new modules (`EdgeSPD`, `LearnableContrast`) built in — stock `pip install ultralytics` will NOT load the v8 checkpoints.

```bash
git clone https://github.com/Sompote/YOLO26-RD.git
cd YOLO26-RD
pip install -e .
```

## Usage

```bash
# Train (n scale; copy the YAML to yolo26s-rd-v8.yaml etc. to select other scales)
yolo detect train model=models/yolo26-rd-v8.yaml data=your_road_damage.yaml \
     imgsz=1280 epochs=300 batch=16

# Validate
yolo detect val model=runs/detect/train/weights/best.pt data=your_road_damage.yaml

# Predict
yolo predict model=runs/detect/train/weights/best.pt source=path/to/images
```

Classes (`nc: 3`): `alligator`, `crack`, `patching` — edit `nc` and your dataset YAML to match.

### Recommended training recipe

- **High input resolution** (`imgsz=1280`) — source imagery is ~3400 px wide; the P2 head only pays off if hairline cracks stay ≥ 2 px.
- **CLAHE preprocessing** for the grayscale, low-contrast imagery (`clipLimit=3, tileGridSize=(8,8)`), or install `albumentations` and raise CLAHE probability — complements the LearnableContrast stem.
- **Oversample crack-heavy images** (duplicate their train-list entries) — Ultralytics has no per-class loss weights.

## Ablation protocol (paper)

| Run | Config |
|---|---|
| A0 | v6 baseline |
| A1 | v7 (SPD + A2C2f + P2 + BiFPN, known components) |
| A2 | v7 with Focus → **EdgeSPD** |
| A3 | v7 + **LearnableContrast** stem |
| A4 | **v8 full** (A2 + A3) |
| A5 | A4 + CLAHE preprocessing |

## New module reference

Both modules are registered in `ultralytics/nn/tasks.py` and usable in any model YAML:

```yaml
- [-1, 1, LearnableContrast, [3]]   # input stem; args: [channels, hidden_width, tile_grid]
- [-1, 1, EdgeSPD, [256, 3]]        # downsample ↓2; args: [out_channels, fusion_kernel]
```

Implementation: [`ultralytics/nn/modules/conv.py`](ultralytics/nn/modules/conv.py) (classes `EdgeSPD`, `LearnableContrast`).

## License

AGPL-3.0, inherited from the Ultralytics codebase this fork is built on.
