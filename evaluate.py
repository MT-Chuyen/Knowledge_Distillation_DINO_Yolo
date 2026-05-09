"""mAP@0.5 evaluation + per-model decoding.

Two decoders, dispatched by inspecting the model output:

  * Teacher  (`HeadOutput`)     — per-cell decoder + per-class NMS, same as before.
  * Student  (Ultralytics YOLO) — Ultralytics' `non_max_suppression` on the
                                   `[B, 4+nc, A]` decoded prediction tensor.

Both produce per-image `(boxes_xyxy_normalized, scores, classes)` tuples that
go into the same VOC-style mAP computation.

CLI:
    python evaluate.py --model teacher --weights runs/teacher/best.pt
    python evaluate.py --model student --weights runs/student_kd/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from boxes import box_iou_xyxy, cxcywh_to_xyxy, nms
from data import YoloAnimalDataset, collate_fn
from models import HeadOutput


# ---------------------------------------------------------------------------
# Teacher decoder — dense head outputs
# ---------------------------------------------------------------------------


def decode_teacher_predictions(
    out: HeadOutput,
    conf_threshold: float,
    nms_iou: float,
    max_per_image: int = 300,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    obj_prob = torch.sigmoid(out.obj_logits)
    cls_prob = torch.softmax(out.cls_logits, dim=-1)
    cls_score, cls_idx = cls_prob.max(dim=-1)
    score = obj_prob * cls_score
    box_xyxy = cxcywh_to_xyxy(out.box).clamp(0.0, 1.0)

    B = score.shape[0]
    results = []
    for b in range(B):
        s = score[b].reshape(-1)
        c = cls_idx[b].reshape(-1)
        bx = box_xyxy[b].reshape(-1, 4)

        keep = s >= conf_threshold
        s, c, bx = s[keep], c[keep], bx[keep]

        if s.numel() > 0:
            offsets = c.float().unsqueeze(-1) * 2.0
            keep_idx = nms(bx + offsets, s, iou_thr=nms_iou)
            if keep_idx.numel() > max_per_image:
                topk = s[keep_idx].topk(max_per_image).indices
                keep_idx = keep_idx[topk]
            s, c, bx = s[keep_idx], c[keep_idx], bx[keep_idx]

        results.append((bx.detach().cpu(), s.detach().cpu(), c.detach().cpu()))
    return results


# ---------------------------------------------------------------------------
# Student decoder — Ultralytics YOLO output
# ---------------------------------------------------------------------------


def decode_yolo_predictions(
    yolo_out,
    img_size: int,
    conf_threshold: float,
    nms_iou: float,
    num_classes: int,
    max_per_image: int = 300,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Decode YOLOv11 eval-mode output and apply per-class NMS.

    In eval mode, DetectionModel returns either:
      - a tuple (decoded_tensor, aux_dict) where decoded_tensor is
        [B, 4+nc, num_anchors] with boxes in *pixel* xyxy coords across all scales
        (4116 anchors at 448 input), or
      - the same tensor alone in some versions.
    Class scores are sigmoid-activated; we threshold on `conf_threshold`,
    pick the top class per anchor, then apply per-class NMS.
    """
    if isinstance(yolo_out, (tuple, list)):
        preds = yolo_out[0]
    else:
        preds = yolo_out
    # preds: [B, 4 + nc, A]
    boxes_xyxy_px = preds[:, :4, :]                  # [B, 4, A]
    cls_scores = preds[:, 4:4 + num_classes, :]       # [B, nc, A]

    B, _, A = preds.shape
    results = []
    for b in range(B):
        b_xyxy = boxes_xyxy_px[b].transpose(0, 1).contiguous()    # [A, 4]
        b_scores = cls_scores[b]                                  # [nc, A]
        score, cls = b_scores.max(dim=0)                          # [A], [A]

        keep = score >= conf_threshold
        b_xyxy = b_xyxy[keep]
        score = score[keep]
        cls = cls[keep]

        if score.numel() > 0:
            offsets = cls.float().unsqueeze(-1) * (img_size + 1.0)
            keep_idx = nms(b_xyxy + offsets, score, iou_thr=nms_iou)
            if keep_idx.numel() > max_per_image:
                topk = score[keep_idx].topk(max_per_image).indices
                keep_idx = keep_idx[topk]
            b_xyxy = b_xyxy[keep_idx]
            score = score[keep_idx]
            cls = cls[keep_idx]

        # Normalize from pixel coords to [0, 1] for the mAP evaluator.
        b_xyxy_norm = (b_xyxy / float(img_size)).clamp(0.0, 1.0)
        results.append((b_xyxy_norm.detach().cpu(),
                        score.detach().cpu(),
                        cls.detach().cpu()))
    return results


# ---------------------------------------------------------------------------
# AP / mAP
# ---------------------------------------------------------------------------


def _ap_from_matches(matched: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if len(matched) == 0:
        return 0.0 if n_gt > 0 else float("nan")
    order = np.argsort(-scores)
    matched = matched[order]
    tp = np.cumsum(matched == 1)
    fp = np.cumsum(matched == 0)
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)

    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map_50(preds_per_image, gts_per_image, num_classes: int, iou_thr: float = 0.5):
    aps = {}
    for cls_id in range(num_classes):
        all_scores: List[float] = []
        all_matched: List[int] = []
        n_gt = 0
        for (bx, sc, cl), gt in zip(preds_per_image, gts_per_image):
            mask_p = cl == cls_id
            p_box, p_sc = bx[mask_p], sc[mask_p]

            if gt.numel():
                gt_mask = gt[:, 0].long() == cls_id
                gt_cls = gt[gt_mask]
                g_xyxy = cxcywh_to_xyxy(gt_cls[:, 1:5]) if gt_cls.numel() else torch.zeros((0, 4))
            else:
                g_xyxy = torch.zeros((0, 4))

            n_gt += g_xyxy.shape[0]
            if p_box.numel() == 0:
                continue
            if g_xyxy.numel() == 0:
                all_scores.extend(p_sc.tolist())
                all_matched.extend([0] * p_sc.numel())
                continue

            ious = box_iou_xyxy(p_box, g_xyxy).numpy()
            order = np.argsort(-p_sc.numpy())
            taken = np.zeros(g_xyxy.shape[0], dtype=bool)
            for i in order:
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_thr and not taken[j]:
                    taken[j] = True
                    all_matched.append(1)
                else:
                    all_matched.append(0)
                all_scores.append(float(p_sc[i].item()))

        ap = _ap_from_matches(np.array(all_matched), np.array(all_scores), n_gt)
        if not np.isnan(ap):
            aps[cls_id] = ap
    if not aps:
        return 0.0, {}
    return float(np.mean(list(aps.values()))), aps


# ---------------------------------------------------------------------------
# Predict over a dataloader (dispatches by model type)
# ---------------------------------------------------------------------------


def predict(model, loader, device, conf_threshold, nms_iou, img_size, num_classes, amp=True):
    model.eval()
    preds_all, gts_all = [], []
    autocast_ctx = torch.autocast(device_type=device.type, enabled=amp)
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with autocast_ctx:
                out = model(imgs)
            if isinstance(out, HeadOutput):
                preds = decode_teacher_predictions(out, conf_threshold, nms_iou)
            else:
                preds = decode_yolo_predictions(out, img_size, conf_threshold, nms_iou, num_classes)
            preds_all.extend(preds)
            gts_all.extend(targets)
    return preds_all, gts_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_model(model_kind: str, cfg: dict, device):
    from models import DinoV2Teacher, YoloV11sStudent
    if model_kind == "teacher":
        return DinoV2Teacher(
            num_classes=cfg["data"]["num_classes"],
            grid_size=cfg["data"]["grid_size"],
            embed_dim=cfg["teacher"]["embed_dim"],
            head_hidden=cfg["teacher"]["head_hidden"],
            dinov2_name=cfg["teacher"]["dinov2_name"],
            lora_r=cfg["teacher"]["lora_r"],
            lora_alpha=cfg["teacher"]["lora_alpha"],
            lora_dropout=cfg["teacher"]["lora_dropout"],
            freeze_backbone=cfg["teacher"]["freeze_backbone"],
        ).to(device)
    if model_kind == "student":
        return YoloV11sStudent(
            num_classes=cfg["data"]["num_classes"],
            weights=cfg["student"]["yolo_weights"],
            kd_feat_layer_idx=cfg["student"]["kd_feat_layer_idx"],
            loss_box=cfg["yolo_loss"]["box"],
            loss_cls=cfg["yolo_loss"]["cls"],
            loss_dfl=cfg["yolo_loss"]["dfl"],
            pretrained=cfg["student"]["pretrained"],
        ).to(device)
    raise ValueError(model_kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=["teacher", "student"], required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = YoloAnimalDataset(
        root=cfg["data"]["root"], split=args.split,
        img_size=cfg["data"]["img_size"], num_classes=cfg["data"]["num_classes"],
        augment=False,
    )
    loader = DataLoader(
        ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Eval split={args.split} | samples={len(ds)}")

    model = _build_model(args.model, cfg, device)
    sd = torch.load(args.weights, map_location=device)
    if args.model == "teacher" and "trainable" in sd:
        model.load_trainable_state_dict(sd["trainable"])
    else:
        model.load_state_dict(sd["model"] if "model" in sd else sd)

    preds, gts = predict(
        model, loader, device,
        conf_threshold=cfg["train"]["conf_threshold"],
        nms_iou=cfg["train"]["nms_iou"],
        img_size=cfg["data"]["img_size"],
        num_classes=cfg["data"]["num_classes"],
        amp=cfg["train"]["amp"],
    )
    map50, per_cls = compute_map_50(preds, gts, num_classes=cfg["data"]["num_classes"])
    print(f"\nmAP@0.5 = {map50:.4f}")
    for cls_id, ap in sorted(per_cls.items()):
        print(f"  class {cls_id:2d}: AP={ap:.4f}")


if __name__ == "__main__":
    main()
