"""YOLO multi-directory dataset.

The dataset on disk is laid out per animal class folder:
    <root>/<animal>/{train,val,test}/{images,labels}
Labels are YOLO format: one row per bbox = `class_id cx cy w h` (normalized).

We merge every animal directory under <root> for the requested split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# DINOv2 / ImageNet normalization. Both teacher and student use the same input pipeline so
# distillation operates on identical pixel values.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass
class Sample:
    img_path: Path
    label_path: Path


def _scan_split(root: Path, split: str) -> List[Sample]:
    samples: List[Sample] = []
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        img_dir = animal_dir / split / "images"
        lbl_dir = animal_dir / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            samples.append(Sample(img, lbl))
    return samples


def _load_yolo_labels(path: Path, num_classes: int) -> torch.Tensor:
    """Read a YOLO label file. Returns a [N, 5] tensor of (cls, cx, cy, w, h).

    Boxes with class >= num_classes or zero area are skipped.
    """
    rows: List[List[float]] = []
    if path.stat().st_size == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
            if cls < 0 or cls >= num_classes:
                continue
            if w <= 0 or h <= 0:
                continue
            rows.append([cls, cx, cy, w, h])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


class YoloAnimalDataset(Dataset):
    """All YOLO labels under root/<animal>/<split>/labels merged into one dataset.

    Returns
    -------
    img : FloatTensor [3, H, W] normalized
    targets : FloatTensor [N, 5] (cls, cx, cy, w, h) — variable N per image
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int,
        num_classes: int,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.samples = _scan_split(self.root, split)
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} for split={split}")

        # Resize-only pipeline: keep boxes in normalized coords so the resize is a no-op for them.
        # Random horizontal flip is applied manually so we can flip the boxes too.
        self.augment = augment
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        targets = _load_yolo_labels(s.label_path, self.num_classes)

        if self.augment and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if targets.numel():
                targets[:, 1] = 1.0 - targets[:, 1]  # flip cx

        img_t = self.to_tensor(img)
        return img_t, targets


def collate_fn(batch):
    """Stack images, keep targets as a list of variable-length tensors."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


def build_targets(
    targets_list: List[torch.Tensor],
    grid_size: int,
    num_classes: int,
    device: torch.device,
):
    """YOLO-v1-style center-cell assignment.

    Returns
    -------
    obj_t : [B, G, G]                — 1 where a GT box center falls, else 0
    cls_t : [B, G, G, num_classes]   — one-hot class at positive cells, 0 elsewhere
    box_t : [B, G, G, 4]             — (cx, cy, w, h) in image-relative coords at positive cells
    pos_mask : [B, G, G] bool        — same as obj_t > 0, returned for convenience
    """
    B = len(targets_list)
    G = grid_size

    obj_t = torch.zeros((B, G, G), device=device)
    cls_t = torch.zeros((B, G, G, num_classes), device=device)
    box_t = torch.zeros((B, G, G, 4), device=device)

    for b, t in enumerate(targets_list):
        if t.numel() == 0:
            continue
        t = t.to(device)
        cls = t[:, 0].long()
        cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
        gx = (cx * G).long().clamp(0, G - 1)
        gy = (cy * G).long().clamp(0, G - 1)

        # If two boxes claim the same cell, the later one overrides — fine for a tutorial-grade
        # detector; SimOTA-style multi-cell assignment would be a heavier upgrade.
        obj_t[b, gy, gx] = 1.0
        cls_t[b, gy, gx] = 0.0
        cls_t[b, gy, gx, cls] = 1.0
        box_t[b, gy, gx, 0] = cx
        box_t[b, gy, gx, 1] = cy
        box_t[b, gy, gx, 2] = w
        box_t[b, gy, gx, 3] = h

    return obj_t, cls_t, box_t, obj_t > 0
