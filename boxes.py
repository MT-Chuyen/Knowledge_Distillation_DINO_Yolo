"""Bounding-box utilities: format conversion, IoU/GIoU, and NMS.

Boxes are torch tensors. Two layouts are used:
  - cxcywh: (cx, cy, w, h)   — normalized to [0, 1] (image-relative).
  - xyxy : (x1, y1, x2, y2)  — same normalization unless stated.
"""

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. a:[N,4], b:[M,4] -> [N,M]."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def giou_loss_cxcywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss between matched boxes (same shape, in cxcywh)."""
    p = cxcywh_to_xyxy(pred)
    t = cxcywh_to_xyxy(target)

    area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
    area_t = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)

    lt = torch.max(p[..., :2], t[..., :2])
    rb = torch.min(p[..., 2:], t[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1e-9)

    enc_lt = torch.min(p[..., :2], t[..., :2])
    enc_rb = torch.max(p[..., 2:], t[..., 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-9)
    return 1.0 - giou  # loss


def nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Standard NMS. Returns indices of kept boxes."""
    if boxes_xyxy.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes_xyxy.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou_xyxy(boxes_xyxy[i:i + 1], boxes_xyxy[rest]).squeeze(0)
        order = rest[ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)
