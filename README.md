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

All variants below were trained under one identical recipe (640², batch 32, 120 epochs, MuSGD,
mosaic 0.5) against a **recipe-matched stock YOLO26-s control** (val 0.773 / test 0.709 mAP50).
Scale is chosen by filename: copy e.g. `yolo26-rd-v8b.yaml` → `yolo26s-rd-v8b.yaml` to train
scale *s*. Test = untouched 551-image split, evaluated once per model.

| Version | Config | Description | val mAP50 | test mAP50 |
|---|---|---|---|---|
| v3–v6 | (training server) | Prior baselines; best v6 = stock YOLO26-s, unrecorded recipe | 0.771 | — |
| v7 | [`models/yolo26-rd-v7.yaml`](models/yolo26-rd-v7.yaml) | Focus(SPD) + A2C2f + P2 head + BiFPN skips — module-free ablation base | — | — |
| v7b | [`models/yolo26-rd-v7b.yaml`](models/yolo26-rd-v7b.yaml) | v7 with the P2 *detection* level removed (module-free control for v8b) | — | — |
| v8 | [`models/yolo26-rd-v8.yaml`](models/yolo26-rd-v8.yaml) | Full CE-SPD: v7 + LearnableContrast stem + EdgeSPD | 0.759 | 0.728 |
| **v8b** | [`models/yolo26-rd-v8b.yaml`](models/yolo26-rd-v8b.yaml) | **YOLO26-RD (recommended)**: v8 with the P2 detection level removed — P2 features stay fused; Detect(P3,P4,P5) | **0.787** | **0.737** |
| v8a | [`models/yolo26-rd-v8a.yaml`](models/yolo26-rd-v8a.yaml) | v8 with genuine area attention at head-P3 (`a2=True, area=4`) | 0.763 | 0.708 |
| v8c | [`models/yolo26-rd-v8c.yaml`](models/yolo26-rd-v8c.yaml) | v8 with EdgeSPD also at P1/2→P2/4 (earliest downsample) | 0.782 | 0.725 |
| v8e | [`models/yolo26-rd-v8e.yaml`](models/yolo26-rd-v8e.yaml) | v8b + v8c combined — *interferes; worse than either parent* | 0.766 | 0.703 |
| v8b-dfl | [`models/yolo26-rd-v8b-dfl.yaml`](models/yolo26-rd-v8b-dfl.yaml) | v8b with distribution box regression restored (`reg_max` 16) — *no gain measured* | 0.759 | — |
| v8b-strip | [`models/yolo26-rd-v8b-strip.yaml`](models/yolo26-rd-v8b-strip.yaml) | v8b + zero-init directional **strip context** (dw 1×9 + 9×1) at head-P3 — *under evaluation* | — | — |

The v8e / v8b-dfl rows are kept deliberately: they are measured negative results, and the val→test
drops across the table (e.g. v8c 0.782→0.725) are why no variant is recommended from validation
alone. The strongest configuration measured to date is **v8b trained with RandomRotate90 +
geometry-targeted oversampling** (val 0.799 / test 0.737; the same augmentation lifts the stock
control to test 0.746 — see the paper's F3 arms before claiming an architecture margin).

Baseline validation results (v3–v6, all classes):

| | v3 | v4 | v5 | v6 |
|---|---|---|---|---|
| mAP50 | 0.746 | 0.717 | 0.750 | **0.771** |
| mAP50-95 | 0.437 | 0.440 | 0.448 | **0.467** |
| Precision | 0.769 | 0.801 | 0.757 | **0.806** |
| Recall | 0.677 | 0.671 | 0.677 | **0.713** |

Per-class (v6): alligator 0.843 · crack 0.719 · patching 0.751 — **crack is the bottleneck class** that CE-SPD targets.

## How to use

### 1. Install

This repository is a fork of Ultralytics with the new modules (`EdgeSPD`, `LearnableContrast`,
`StripC3k2`, `StripA2C2f`) built in — stock `pip install ultralytics` will **not** load the
v8-family checkpoints or YAMLs.

```bash
git clone https://github.com/Sompote/YOLO26-RD.git
cd YOLO26-RD
pip install -e .
pip install albumentations   # optional, needed only for the RandomRotate90 recipe below
```

Verify the fork (not PyPI ultralytics) is active:

```bash
python -c "from ultralytics.nn.modules import EdgeSPD, LearnableContrast, StripC3k2; print('fork OK')"
```

### 2. Prepare your dataset

Standard YOLO detection layout, with a `data.yaml`:

```yaml
path: /path/to/dataset        # root
train: train/images
val: valid/images
test: test/images
nc: 3
names: ['alligator crack', 'crack', 'patching']
```

Labels are one `class cx cy w h` line per box (normalized), in `*/labels/` mirroring `*/images/`.
Edit `nc`/`names` to your classes — the model YAMLs read `nc` from your data file at train time.

### 3. Pick a variant and a scale

The recommended variant is **v8b (YOLO26-RD)**. The model **scale is read from the YAML
filename** (`n`/`s`/`m`/`l`/`x`); a file without a scale letter silently trains scale `n` with a
warning, so always copy to a scale-lettered name first:

```bash
cp models/yolo26-rd-v8b.yaml models/yolo26s-rd-v8b.yaml   # scale s (12.4M params) — recommended
```

### 3b. Using each variant

Every variant trains with the same command — only the YAML changes. Copy to a scale-lettered
name, then substitute it for `MODEL` in §4:

```bash
# recommended model — YOLO26-RD: P2 features fused, Detect(P3,P4,P5)
cp models/yolo26-rd-v8b.yaml models/yolo26s-rd-v8b.yaml            # 12.42M, test mAP50 0.737

# full CE-SPD as originally published — adds the P2 detection level back
cp models/yolo26-rd-v8.yaml models/yolo26s-rd-v8.yaml              # 11.97M, test 0.728; ~8% slower than v8b

# genuine area attention at head-P3 (a2=True, area=4)
cp models/yolo26-rd-v8a.yaml models/yolo26s-rd-v8a.yaml            # 11.99M, test 0.708; +22% epoch time

# EdgeSPD also at the earliest downsample (P1/2->P2/4)
cp models/yolo26-rd-v8c.yaml models/yolo26s-rd-v8c.yaml            # 12.02M, test 0.725; best val mAP50-95

# v8b + v8c combined — measured NEGATIVE result, kept for reproducibility only
cp models/yolo26-rd-v8e.yaml models/yolo26s-rd-v8e.yaml            # 12.47M, test 0.703 — do not use

# v8b with distribution box regression (reg_max 16) — no gain measured on this dataset
cp models/yolo26-rd-v8b-dfl.yaml models/yolo26s-rd-v8b-dfl.yaml    # 13.12M; loss shows dfl_loss instead of l1_loss

# v8b + zero-init directional strip context at head-P3 — experimental, under evaluation
cp models/yolo26-rd-v8b-strip.yaml models/yolo26s-rd-v8b-strip.yaml # +2.3k params over v8b

# ablation baselines (no LearnableContrast / EdgeSPD; Focus/SPD instead)
cp models/yolo26-rd-v7.yaml  models/yolo26s-rd-v7.yaml             # with P2 detection level
cp models/yolo26-rd-v7b.yaml models/yolo26s-rd-v7b.yaml            # without (module-free control for v8b)
```

Variant-specific notes:

- **v8b / v8** — checkpoints are *not* interchangeable: v8 has four detection scales (P2–P5),
  v8b three (P3–P5). Fine-tuning one from the other transfers backbone/neck weights only.
- **v8b-dfl** — `reg_max: 16` changes the head's output layout; its checkpoints load only with
  this YAML family, and training prints `dfl_loss` in place of `l1_loss` (expected).
- **v8b-strip** — starts *exactly* equal to v8b (zero-initialized gate), so short runs that show
  no difference from v8b are behaving correctly; the strip context only acts once the gate learns.
- **v8e** — retained so the published negative result can be reproduced; it underperforms both of
  its parents on both validation and test.
- **v7 / v7b** — the module-free ablation pair; use these to measure what LearnableContrast +
  EdgeSPD contribute inside this architecture on your own data.
- All variants share the same dataset format, recipe flags, validation and export commands (§4–§6);
  none needs NMS at deployment.

### 4. Train

CLI (measured recipe for this dataset — 640², reduced mosaic, vertical flip):

```bash
yolo detect train model=models/yolo26s-rd-v8b.yaml data=data.yaml \
     imgsz=640 epochs=120 batch=32 mosaic=0.5 close_mosaic=30 flipud=0.5 cos_lr=True
```

Python API — required for the full recipe, because the RandomRotate90 augmentation (the single
largest measured gain on this dataset: +3.7 test mAP50 on the stock control) is passed as an
albumentations list:

```python
import albumentations as A
from ultralytics import YOLO

model = YOLO("models/yolo26s-rd-v8b.yaml")
model.train(
    data="data.yaml",
    imgsz=640, epochs=120, batch=32,
    mosaic=0.5, close_mosaic=30, flipud=0.5, cos_lr=True,
    augmentations=[A.RandomRotate90(p=0.5)],   # label-exact for axis-aligned boxes
)
```

Notes:
- Training is **from scratch** by default (a `.yaml` model ignores `pretrained=True`); pass
  `pretrained=path/to/weights.pt` explicitly to warm-start.
- Resume an interrupted run with `yolo detect train resume model=runs/detect/train/weights/last.pt`
  (re-pass `augmentations=` — it is not restored automatically).
- To oversample rare/hard classes, point `train:` in `data.yaml` at a `.txt` file of absolute image
  paths and duplicate lines (duplicates are kept); labels resolve by the `images/` → `labels/`
  path swap. Alternatively set `cls_pw` (this fork supports inverse-frequency^cls_pw per-class
  loss weighting; 0.0 disables).

### 5. Validate and test

```bash
# validation split (drives best.pt selection: fitness = 0.9*mAP50-95 + 0.1*mAP50)
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640

# held-out test split — report this, once, after all decisions are frozen
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640 split=test
```

**Always validate at the training `imgsz`.** Measured on this dataset, evaluating a 640-trained
model at 800/960 *reduces* mAP50 (0.787 → 0.758 → 0.701). Note that test-time augmentation
(`augment=True`) is a no-op for these NMS-free `end2end` models.

### 6. Predict and export

```bash
yolo predict model=runs/detect/train/weights/best.pt source=path/to/images conf=0.25
yolo export  model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

The head is NMS-free (one-to-one, `reg_max: 1`), so exported models need no NMS post-processing.

### Recommended training recipe (evidence-based)

- **Train and validate at the same `imgsz`** — see §5. We found no benefit from 1280 training; box
  geometry (98.6% of instances above the COCO-small threshold at 640) does not require it.
- **Reduce mosaic** (`mosaic=0.5`, generous `close_mosaic`) — at `mosaic=1.0` accuracy peaked near
  epoch 87 of 400 and then declined; mosaic tiling truncates near-full-frame crack boxes.
- **RandomRotate90** — corrects longitudinal/transverse orientation imbalance (see §4).
- **CLAHE preprocessing** (`clipLimit=3, tileGridSize=(8,8)`) for grayscale, low-contrast imagery —
  complements the LearnableContrast stem.
- **Oversample crack-heavy images** and/or set `cls_pw` (see §4).

## Ablation protocol (paper)

Completed arms, all under the identical recipe (see the paper for full val/test tables):

| Run | Config | Outcome (test mAP50) |
|---|---|---|
| baseline | stock YOLO26-s, recipe-matched | 0.709 |
| A1 | v8 (adds P2 detection level back to v8b) | 0.728 |
| A2 | v8c (EdgeSPD at earliest downsample) | 0.725 |
| A3 | v8a (genuine head-P3 area attention) | 0.708 |
| A4 | v8e (v8b + v8c combined) | 0.703 — negative |
| A5 | stock + EdgeSPD | 0.706 — val-only gain |
| A6 | stock + EdgeSPD + LearnableContrast | 0.716 |
| — | **v8b (YOLO26-RD, proposed)** | **0.737** |

## New module reference

All four modules are registered in `ultralytics/nn/tasks.py` and usable in any model YAML:

```yaml
- [-1, 1, LearnableContrast, [3]]        # input stem; args: [channels, hidden_width, tile_grid]
- [-1, 1, EdgeSPD, [256, 3]]             # downsample ↓2; args: [out_channels, fusion_kernel]
- [-1, 2, StripC3k2, [256, True]]        # C3k2 + zero-init dw 1×9/9×1 strip context (same args as C3k2, optional k)
- [-1, 2, StripA2C2f, [256, False, -1]]  # A2C2f + the same strip context (same args as A2C2f, optional k)
```

The strip variants add ~2.3k parameters and start exactly equal to their parent block (scalar gate
zero-initialized), so they can be dropped into any existing config without changing initial behavior.

Implementation: [`ultralytics/nn/modules/conv.py`](ultralytics/nn/modules/conv.py) (`EdgeSPD`,
`LearnableContrast`) and [`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py)
(`StripC3k2`, `StripA2C2f`).

## License

AGPL-3.0, inherited from the Ultralytics codebase this fork is built on.
