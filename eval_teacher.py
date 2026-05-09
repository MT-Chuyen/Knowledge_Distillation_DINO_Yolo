"""Independent multi-label evaluation of the trained teacher.

Treats the detector as a multi-label classifier: per image, the predicted set
of class IDs (from detections passing NMS + confidence threshold) is compared
to the GT set of class IDs (unique class ids in the YOLO label file).

Reports:
  * Exact Match Ratio                       — fraction of images with pred_set == gt_set
  * Micro / Macro {Precision, Recall, F1}   — over per-class binary indicators
  * Per-class table (Support, P, R, F1)     — only classes with support > 0

Class name mapping is auto-derived: for each class_id, we pick the dataset
directory under `root/` whose train labels mention that class most often.
This works because each directory is named after one species.

Usage:
    python eval_teacher.py
    python eval_teacher.py --split test --conf 0.3
    python eval_teacher.py --weights runs/teacher/best.pt --names class_names.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import YoloAnimalDataset, collate_fn
from evaluate import predict
from models import DinoV2Teacher


# ---------------------------------------------------------------------------
# Class-name resolution
# ---------------------------------------------------------------------------


def derive_class_names(root: Path, num_classes: int) -> Dict[int, str]:
    """For each class id, pick the dataset folder whose train labels contain it
    most often. Fallback name `class_<id>` if a class id never appears.
    """
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        labels_dir = animal_dir / "train" / "labels"
        if not labels_dir.is_dir():
            continue
        for txt in labels_dir.glob("*.txt"):
            try:
                lines = txt.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cls < num_classes:
                    counts[cls][animal_dir.name] += 1

    names: Dict[int, str] = {}
    for cls in range(num_classes):
        if cls in counts and counts[cls]:
            names[cls] = max(counts[cls].items(), key=lambda kv: kv[1])[0]
        else:
            names[cls] = f"class_{cls}"
    return names


# ---------------------------------------------------------------------------
# Multi-label metric computation
# ---------------------------------------------------------------------------


def build_label_sets(
    preds_per_image: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    gts_per_image: List[torch.Tensor],
) -> Tuple[List[set], List[set]]:
    """Per image: predicted-class set and ground-truth-class set.

    Predictions have already been filtered by `conf_threshold` and NMS in
    `evaluate.predict`, so we just collect the unique class ids that survived.
    """
    pred_sets: List[set] = []
    gt_sets: List[set] = []
    for (_, _, classes), gt in zip(preds_per_image, gts_per_image):
        pred_sets.append(set(int(c) for c in classes.tolist()))
        if gt.numel():
            gt_sets.append(set(int(c) for c in gt[:, 0].long().tolist()))
        else:
            gt_sets.append(set())
    return pred_sets, gt_sets


def multilabel_metrics(pred_sets: List[set], gt_sets: List[set], num_classes: int):
    """Per-class TP/FP/FN/Support + micro/macro aggregates + exact-match ratio."""
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    n_exact = 0
    for ps, gs in zip(pred_sets, gt_sets):
        if ps == gs:
            n_exact += 1
        for c in range(num_classes):
            in_p, in_g = c in ps, c in gs
            if in_p and in_g:
                tp[c] += 1
            elif in_p and not in_g:
                fp[c] += 1
            elif not in_p and in_g:
                fn[c] += 1

    support = tp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    # Micro: pool TP/FP/FN over all classes.
    mtp, mfp, mfn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    micro_p = mtp / (mtp + mfp) if (mtp + mfp) else 0.0
    micro_r = mtp / (mtp + mfn) if (mtp + mfn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    # Macro: average over classes that actually have ground-truth support.
    has_support = support > 0
    if has_support.any():
        macro_p = float(precision[has_support].mean())
        macro_r = float(recall[has_support].mean())
        macro_f1 = float(f1[has_support].mean())
    else:
        macro_p = macro_r = macro_f1 = 0.0

    exact_match_ratio = n_exact / len(pred_sets) if pred_sets else 0.0

    per_class = []
    for c in range(num_classes):
        per_class.append({
            "class_id": c,
            "support": int(support[c]),
            "tp": int(tp[c]),
            "fp": int(fp[c]),
            "fn": int(fn[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
        })

    return {
        "exact_match_ratio": exact_match_ratio,
        "micro": (micro_p, micro_r, micro_f1),
        "macro": (macro_p, macro_r, macro_f1),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(metrics: dict, class_names: Dict[int, str]):
    print(f"Exact Match Ratio: {metrics['exact_match_ratio']:.4f}\n")

    p, r, f = metrics["micro"]
    print("Micro Average:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-Score:  {f:.4f}\n")

    p, r, f = metrics["macro"]
    print("Macro Average:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-Score:  {f:.4f}")

    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60 + "\n")

    print(f"{'Class':<22}{'Support':>9}{'Precision':>11}{'Recall':>11}{'F1-Score':>11}")
    print("-" * 60)

    rows = [
        (class_names.get(c["class_id"], f"class_{c['class_id']}"), c)
        for c in metrics["per_class"]
        if c["support"] > 0
    ]
    rows.sort(key=lambda kv: kv[0].lower())
    for name, c in rows:
        print(f"{name:<22}{c['support']:>9}{c['precision']:>11.4f}{c['recall']:>11.4f}{c['f1']:>11.4f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default=None,
                        help="Teacher checkpoint. Defaults to teacher.ckpt from config.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=None,
                        help="Override conf threshold (defaults to train.conf_threshold).")
    parser.add_argument("--names", default=None,
                        help="Optional YAML mapping {class_id: name}. If omitted, derived from data.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weights or cfg["teacher"]["ckpt"]
    conf = args.conf if args.conf is not None else cfg["train"]["conf_threshold"]

    # Load checkpoint up front so we can prefer its embedded config over the
    # on-disk one — the teacher's grid_size / img_size is fixed by the head
    # buffers it was trained with, and config.yaml may have drifted.
    sd = torch.load(weights, map_location=device, weights_only=False)
    ckpt_cfg = sd.get("config")
    if ckpt_cfg:
        for section in ("data", "teacher"):
            if section in ckpt_cfg:
                cfg.setdefault(section, {}).update(ckpt_cfg[section])

    nc = cfg["data"]["num_classes"]
    root = Path(cfg["data"]["root"])

    # Class names
    if args.names:
        names_raw = yaml.safe_load(Path(args.names).read_text())
        class_names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        class_names = derive_class_names(root, nc)
    print(f"Class names: {class_names}\n")

    # Data
    ds = YoloAnimalDataset(
        root=cfg["data"]["root"], split=args.split,
        img_size=cfg["data"]["img_size"], num_classes=nc, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Eval split={args.split} | samples={len(ds)} | conf_threshold={conf}")

    # Model
    model = DinoV2Teacher(
        num_classes=nc, grid_size=cfg["data"]["grid_size"],
        embed_dim=cfg["teacher"]["embed_dim"],
        head_hidden=cfg["teacher"]["head_hidden"],
        dinov2_name=cfg["teacher"]["dinov2_name"],
        lora_r=cfg["teacher"]["lora_r"],
        lora_alpha=cfg["teacher"]["lora_alpha"],
        lora_dropout=cfg["teacher"]["lora_dropout"],
        freeze_backbone=cfg["teacher"]["freeze_backbone"],
    ).to(device)

    model.load_trainable_state_dict(sd["trainable"])
    print(f"Loaded teacher checkpoint: {weights}  (epoch={sd.get('epoch')}, mAP={sd.get('map50')})\n")

    preds_all, gts_all = predict(
        model, loader, device,
        conf_threshold=conf,
        nms_iou=cfg["train"]["nms_iou"],
        img_size=cfg["data"]["img_size"],
        num_classes=nc,
        amp=cfg["train"]["amp"],
    )

    pred_sets, gt_sets = build_label_sets(preds_all, gts_all)
    metrics = multilabel_metrics(pred_sets, gt_sets, num_classes=nc)
    print_report(metrics, class_names)


if __name__ == "__main__":
    main()
