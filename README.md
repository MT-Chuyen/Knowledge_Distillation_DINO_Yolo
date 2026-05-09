# KD-animal-detection

Knowledge distillation for object detection on the 20-class YOLO animal dataset
at `/home/chuyenmt/code/full_animal`. Inspired by the
[PyTorch Knowledge Distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html),
adapted from CIFAR-10 classification to YOLO-format detection with a frozen
DINOv2 teacher and an Ultralytics YOLOv11s student.

## Pipeline

```
                            +------------------------+
                            |  Frozen DINOv2-L/14    |
   train_teacher.py  ---->  |  + LoRA adapters       |  ---->  teacher.pt
   (LoRA + det head)        |  + dense det head      |
                            +-----------+------------+
                                        |
                                        |  (frozen during student training:
                                        |   feature MSE + cosine on pooled features)
                                        v
                            +------------------------+
   train_student.py  ---->  |  YOLOv11s (Ultralytics)|  ---->  student_baseline.pt
   --mode {baseline, kd}    |  + cls head re-init    |         student_kd.pt
                            |    for 20 classes      |
                            +------------------------+
```

## Image size

Picked `img_size = 448` so a single resolution works for both backbones:

| | factor | grid |
|---|---|---|
| DINOv2 ViT-L/14 patch size | 14 | 32 × 32 |
| YOLOv11 max stride          | 32 | P3=56, P4=28, P5=14 |

448 is the smallest size divisible by both 14 and 32. Larger choices (672, 896)
work too but make DINOv2 attention much slower.

## Models

| Role | Backbone | Params (trainable / total) | Output |
|---|---|---|---|
| Teacher | DINOv2 ViT-L/14 + LoRA + dense head | ~3.2M / 307M | 32×32 grid, 25 ch |
| Student | YOLOv11s (COCO pretrained, head reinit) | ~9.4M / 9.4M | 3 scales, DFL head |

Teacher head channels per cell = `4 (cx,cy,w,h) + 1 (obj) + 20 (cls)`.
Student is anchor-free with DFL (`reg_max=16`).

## Knowledge-distillation signals

Two of the three tutorial mechanisms map cleanly:

| Tutorial mechanism | Adapted to detection? | Notes |
|---|---|---|
| #1 logit KD (softmax soft-targets, T) | **No** | YOLOv11 multi-scale anchor-free head ≠ teacher dense head; logits don't align spatially or in semantics. |
| #2 cosine on hidden representation | **Yes** | `CosineEmbeddingLoss` on globally-pooled features (after the regressor lifts student to teacher's channel count). |
| #3 FitNets / regressor MSE | **Yes** | 1×1 conv (256 → 1024) + bilinear interp (28→32) projects YOLOv11s P4 (post-neck, layer 19) into teacher feature space, then MSE. |

Mechanism #3 is the primary KD signal; #2 acts as a coarse global regulariser.
Weights / temperature in `kd:` of `config.yaml`.

## Files

| File | What it does |
|---|---|
| `config.yaml` | Single config (paths, img_size, model size, KD weights, YOLO loss gains). |
| `data.py` | Multi-dir YOLO dataset; YOLO-v1 center-cell target builder for the teacher. |
| `boxes.py` | cxcywh↔xyxy, IoU, GIoU loss, NMS. |
| `models.py` | `DinoV2Teacher` (DINOv2 + LoRA + dense head), `YoloV11sStudent` (Ultralytics wrapper with hooked KD feature), `FeatureRegressor` (1×1 conv + spatial interp). |
| `losses.py` | `DetectionLoss` (teacher), `YoloLossAdapter` (wraps Ultralytics `v8DetectionLoss`), `feature_kd_loss`, `cosine_kd_loss`. |
| `train_teacher.py` | Stage 1: fine-tune teacher LoRA + head. Saves only trainable weights. |
| `train_student.py` | Stage 2: trains YOLOv11s. `--mode baseline` (no teacher) or `--mode kd` (feature + cosine KD). |
| `evaluate.py` | mAP@0.5 evaluator. Dispatches between teacher dense head and YOLO multi-scale output. |

## How to run

```bash
# 0. install deps once
/home/chuyenmt/.venv/bin/pip install -r requirements.txt

# 1. fine-tune the teacher (LoRA + head only — DINOv2 frozen)
/home/chuyenmt/.venv/bin/python train_teacher.py --config config.yaml

# 2a. baseline student (YOLOv11s, no teacher) — same supervised loss as Ultralytics' own training
/home/chuyenmt/.venv/bin/python train_student.py --mode baseline

# 2b. student with KD (uses teacher checkpoint from step 1)
/home/chuyenmt/.venv/bin/python train_student.py --mode kd \
    --teacher-ckpt runs/teacher/best.pt

# 3. evaluate any saved checkpoint on val (or test) split
/home/chuyenmt/.venv/bin/python evaluate.py --model teacher --weights runs/teacher/best.pt          --split val
/home/chuyenmt/.venv/bin/python evaluate.py --model student --weights runs/student_baseline/best.pt --split val
/home/chuyenmt/.venv/bin/python evaluate.py --model student --weights runs/student_kd/best.pt       --split val
```

## Why these choices

- **Teacher = DINOv2-L + LoRA**. DINOv2-L is a SOTA-tier backbone (Co-DETR + DINOv2
  holds COCO SOTA); on ~20k images full fine-tune would overfit, so LoRA on
  attention `qkv`/`proj` keeps trainable params at ~1% of the model.
- **Student = YOLOv11s pretrained on COCO**. Real-world choice: its
  backbone+neck are already strong; we only re-init the cls layers for 20
  classes. Using `pretrained: false` gives the more "tutorial-pure" random-init
  comparison.
- **Different heads → no logit KD**. Forcing logit KD across DETR-vs-YOLO heads
  is a research project on its own; we keep KD where it's clean (features).
- **One image size for both**. 448 lands DINOv2-L on a 32×32 grid and
  YOLOv11s on P3=56 / P4=28 / P5=14. Layer-19 (post-neck P4, 28×28) is
  the natural KD source — close to the teacher's 32×32, post-neck (so
  enriched), and 256-channel (a clean 1×1 conv lifts to teacher's 1024).

## Caveats / known limitations

- Teacher uses a YOLO-v1-style center-cell matcher. Two GT boxes in the same
  14-pixel cell collide and one wins. Fine for the tutorial-grade teacher;
  swap to FCOS/SimOTA for serious mAP.
- Logit KD is dropped entirely. If you want it, the cleanest path is a
  *YOLO-style* teacher head (3 scales) on top of DINOv2 features + a custom
  multi-scale logit-KD term — significant rewrite.
- The KD feature is pulled from a single layer (layer 19). Multi-layer
  feature distillation (Reviewer KD, Hint Guided distillation) would likely help.
- One teacher pass per student step is the bottleneck. Pre-computing teacher
  feature maps and caching them once would cut student-train wall time ~3×.
- mAP eval is single-IoU (0.5). Add IoU=0.5:0.95 averaging for COCO-style mAP.
