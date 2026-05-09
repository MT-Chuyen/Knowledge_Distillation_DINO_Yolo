"""Supervised + KD losses.

Two supervised paths, one per architecture:

  * `DetectionLoss`    — dense head loss for the **teacher** (BCE for obj/cls,
                          GIoU+L1 for box). YOLO-v1-style center-cell assignment
                          is built externally in `data.build_targets`.
  * `YoloLossAdapter`  — wraps Ultralytics' `v8DetectionLoss` for the
                          **student** (YOLOv11s). Converts our list-of-tensors
                          target format to Ultralytics' batch dict.

KD losses (used only in the student trainer):

  * `feature_kd_loss`  — MSE between teacher feature map and (regressor-projected)
                          student feature map. Tutorial mechanism #3 (FitNets).
  * `cosine_kd_loss`   — CosineEmbeddingLoss on globally-pooled features.
                          Tutorial mechanism #2.

Logit KD (mechanism #1) is dropped — YOLOv11s' multi-scale anchor-free head and
the teacher's single-scale dense head don't share an output layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from boxes import giou_loss_cxcywh
from models import HeadOutput


# ---------------------------------------------------------------------------
# Teacher supervised loss (dense head)
# ---------------------------------------------------------------------------


@dataclass
class DetLossWeights:
    obj: float = 1.0
    cls: float = 1.0
    box_giou: float = 2.0
    box_l1: float = 1.0


class DetectionLoss(nn.Module):
    """Loss for the teacher's dense head.

    Inputs are the head's decoded outputs plus center-cell-assigned targets
    produced by `data.build_targets`.
    """

    def __init__(self, weights: DetLossWeights | None = None):
        super().__init__()
        self.w = weights or DetLossWeights()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, out: HeadOutput, obj_t, cls_t, box_t, pos_mask):
        obj_loss = self.bce(out.obj_logits, obj_t).mean()
        n_pos = pos_mask.sum().clamp(min=1)

        if pos_mask.any():
            cls_loss = self.bce(out.cls_logits[pos_mask], cls_t[pos_mask]).sum() / n_pos
            pred_box = out.box[pos_mask]
            tgt_box = box_t[pos_mask]
            giou_l = giou_loss_cxcywh(pred_box, tgt_box).mean()
            l1_l = F.l1_loss(pred_box, tgt_box)
        else:
            cls_loss = out.cls_logits.sum() * 0.0
            giou_l = out.box.sum() * 0.0
            l1_l = out.box.sum() * 0.0

        total = (
            self.w.obj * obj_loss
            + self.w.cls * cls_loss
            + self.w.box_giou * giou_l
            + self.w.box_l1 * l1_l
        )
        parts = {
            "obj": obj_loss.detach(),
            "cls": cls_loss.detach(),
            "giou": giou_l.detach(),
            "l1": l1_l.detach(),
            "total": total.detach(),
        }
        return total, parts


# ---------------------------------------------------------------------------
# Student supervised loss (Ultralytics adapter)
# ---------------------------------------------------------------------------


def yolo_targets_from_list(
    targets_list: list[torch.Tensor],
    device: torch.device,
) -> dict:
    """Convert our `list of [N, 5] (cls, cx, cy, w, h)` targets into Ultralytics' batch dict.

    Ultralytics expects:
        batch_idx : [M]      image index per box
        cls       : [M, 1]   class id per box
        bboxes    : [M, 4]   normalized cxcywh
    where M = total boxes in the batch.
    """
    batch_idx_list, cls_list, box_list = [], [], []
    for i, t in enumerate(targets_list):
        if t.numel() == 0:
            continue
        n = t.shape[0]
        batch_idx_list.append(torch.full((n,), i, dtype=torch.float32, device=device))
        cls_list.append(t[:, 0].to(device=device, dtype=torch.float32))
        box_list.append(t[:, 1:5].to(device=device, dtype=torch.float32))

    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list)
        cls = torch.cat(cls_list).view(-1, 1)
        bboxes = torch.cat(box_list)
    else:
        batch_idx = torch.zeros(0, device=device)
        cls = torch.zeros(0, 1, device=device)
        bboxes = torch.zeros(0, 4, device=device)

    return {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


class YoloLossAdapter:
    """Thin wrapper around Ultralytics' `v8DetectionLoss`.

    Build once with the underlying DetectionModel; call with predictions and our
    target list per batch.
    """

    def __init__(self, det_model: nn.Module):
        # `init_criterion` reads `det_model.args.box/cls/dfl` weights.
        self.criterion = det_model.init_criterion()

    def __call__(self, preds, targets_list, device):
        batch = yolo_targets_from_list(targets_list, device=device)
        # v8DetectionLoss returns (loss_vec, loss_detach) where loss_vec is the
        # 3-element tensor [box, cls, dfl] already multiplied by batch_size.
        # Ultralytics' trainer sums this; we do the same.
        loss_vec, items = self.criterion(preds, batch)
        return loss_vec.sum(), items


# ---------------------------------------------------------------------------
# Knowledge distillation
# ---------------------------------------------------------------------------


def feature_kd_loss(student_feat_proj: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """MSE — student must already have been passed through the regressor so
    shapes match.
    """
    return F.mse_loss(student_feat_proj, teacher_feat.detach())


def cosine_kd_loss(student_feat_proj: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """CosineEmbeddingLoss on globally-pooled features.

    Tutorial mechanism #2: pool both feature maps to single vectors and push
    the student's vector to align with the teacher's.
    """
    s = F.adaptive_avg_pool2d(student_feat_proj, 1).flatten(1)   # [B, C]
    t = F.adaptive_avg_pool2d(teacher_feat, 1).flatten(1).detach()
    target = torch.ones(s.size(0), device=s.device)
    return F.cosine_embedding_loss(s, t, target)
