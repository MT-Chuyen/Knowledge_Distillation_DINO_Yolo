"""Teacher and student detectors.

Teacher: DINOv2 ViT-L/14 + LoRA (PEFT) + a single-scale dense detection head.
    Backbone is frozen; only LoRA adapters and the head learn. Output is a
    [B, 1024, G, G] feature map (G = img_size / 14) followed by a dense head
    producing per-cell (obj, cls, box) tensors.

Student: Ultralytics YOLOv11s (multi-scale anchor-free detector with DFL).
    Loaded from `yolo11s.pt` (COCO-pretrained), with the cls heads reinitialised
    for the target num_classes. The Ultralytics `v8DetectionLoss` is used for
    the supervised path; KD is applied at the *backbone/neck* feature level via
    a forward hook on a chosen layer.

Because the heads disagree, distillation is feature-level only:
    - FeatureRegressor: 1x1 conv to lift student's channel count to the teacher's,
      followed by bilinear interpolation to match the teacher's spatial grid.
    This mirrors the tutorial's FitNets / "RegressorMSE" experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Teacher head (single-scale, dense)
# ---------------------------------------------------------------------------


@dataclass
class HeadOutput:
    """Per-cell outputs of the teacher dense head."""

    obj_logits: torch.Tensor       # [B, G, G]
    cls_logits: torch.Tensor       # [B, G, G, num_classes]
    box: torch.Tensor              # [B, G, G, 4]  (cx, cy, w, h) in [0, 1]


class DetectionHead(nn.Module):
    """1x1 -> 3x3 -> 1x1 conv stack on a [B, C, G, G] feature map.

    Output channels are `4 (bbox) + 1 (obj) + num_classes (cls)`. Box is
    parameterised relative to the cell:
        cx = (gx + sigmoid(tx)) / G
        cy = (gy + sigmoid(ty)) / G
        w  = sigmoid(tw)
        h  = sigmoid(th)
    """

    def __init__(self, in_channels: int, hidden: int, num_classes: int, grid_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        out_channels = 4 + 1 + num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(32, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.SiLU(inplace=True),
        )
        self.predict = nn.Conv2d(hidden, out_channels, kernel_size=1)

        gy, gx = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", gx)
        self.register_buffer("grid_y", gy)

    def forward(self, feat: torch.Tensor) -> HeadOutput:
        x = self.stem(feat)
        raw = self.predict(x).permute(0, 2, 3, 1)        # [B, G, G, 5+nc]

        tx, ty, tw, th = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        obj_logits = raw[..., 4]
        cls_logits = raw[..., 5:]

        G = self.grid_size
        cx = (self.grid_x + torch.sigmoid(tx)) / G
        cy = (self.grid_y + torch.sigmoid(ty)) / G
        w = torch.sigmoid(tw)
        h = torch.sigmoid(th)
        box = torch.stack([cx, cy, w, h], dim=-1)
        return HeadOutput(obj_logits=obj_logits, cls_logits=cls_logits, box=box)


# ---------------------------------------------------------------------------
# Teacher: DINOv2-L + LoRA + dense head
# ---------------------------------------------------------------------------


class DinoV2Teacher(nn.Module):
    """DINOv2 ViT-L/14 patch tokens -> [B, 1024, G, G] -> det head."""

    def __init__(
        self,
        num_classes: int,
        grid_size: int,
        embed_dim: int = 1024,
        head_hidden: int = 256,
        dinov2_name: str = "dinov2_vitl14_reg",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.dinov2_name = dinov2_name

        backbone = torch.hub.load("facebookresearch/dinov2", dinov2_name, trust_repo=True)
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["qkv", "proj"],
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        self.head = DetectionHead(
            in_channels=embed_dim,
            hidden=head_hidden,
            num_classes=num_classes,
            grid_size=grid_size,
        )

    def _patch_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        patch = feats["x_norm_patchtokens"]
        B, N, C = patch.shape
        G = self.grid_size
        if N != G * G:
            raise RuntimeError(
                f"Expected {G*G} patch tokens (img_size/14 = {G}), got {N}. "
                f"Adjust config.data.img_size or grid_size."
            )
        return patch.transpose(1, 2).reshape(B, C, G, G)

    def forward(self, x: torch.Tensor) -> HeadOutput:
        feat = self._patch_feature_map(x)
        return self.head(feat)

    def forward_with_features(self, x: torch.Tensor):
        feat = self._patch_feature_map(x)
        return self.head(feat), feat   # HeadOutput, [B, embed_dim, G, G]

    def trainable_state_dict(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, sd: dict, strict: bool = False):
        own = dict(self.named_parameters())
        missing = []
        for k, v in sd.items():
            if k not in own:
                missing.append(k)
                continue
            own[k].data.copy_(v.to(own[k].device))
        if strict and missing:
            raise RuntimeError(f"Missing keys: {missing[:5]}...")


# ---------------------------------------------------------------------------
# Student: Ultralytics YOLOv11s wrapper
# ---------------------------------------------------------------------------


class YoloV11sStudent(nn.Module):
    """Wrap an Ultralytics YOLOv11s model so it plays nicely with our trainer.

    Responsibilities:
    1. Load `yolo11s.pt` (COCO-pretrained), then re-init the cls head layers
       for the target `num_classes` while keeping the backbone weights.
    2. Register a forward hook on `kd_feat_layer_idx` so we can grab an
       intermediate feature map for distillation.
    3. Expose three forward modes:
         - `forward(x)`             — Ultralytics default (training: list of
                                       per-scale feature maps; eval: decoded
                                       predictions tensor + raw feature maps).
         - `forward_with_features`  — same as forward, but also returns the
                                       hooked KD feature.
         - `feat_channels`          — channel count of the hooked feature.

    Notes
    -----
    * Ultralytics' DetectionModel uses `model.args` for loss hyperparameters
      (`box`, `cls`, `dfl`). We attach those at construction so callers can
      build the loss with `model.init_criterion()`.
    * The KD feature's spatial size is whatever the chosen layer outputs
      (28x28 at 448 input for layer 19). Spatial alignment with the teacher
      grid is the regressor's job, not ours.
    """

    def __init__(
        self,
        num_classes: int,
        weights: str = "yolo11s.pt",
        kd_feat_layer_idx: int = 19,
        loss_box: float = 7.5,
        loss_cls: float = 0.5,
        loss_dfl: float = 1.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kd_feat_layer_idx = kd_feat_layer_idx

        from types import SimpleNamespace

        from ultralytics import YOLO
        yolo = YOLO(weights)
        self.det_model = yolo.model           # nn.Module DetectionModel
        if not pretrained:
            self.det_model.apply(_reset_weights)

        self._reinit_cls_head(num_classes)
        self.det_model.nc = num_classes
        self.det_model.names = {i: str(i) for i in range(num_classes)}
        self.det_model.args = SimpleNamespace(
            box=loss_box, cls=loss_cls, dfl=loss_dfl,
        )

        self._kd_feat: Optional[torch.Tensor] = None
        self._register_kd_hook()

        # Run a dummy forward to (a) populate self._kd_feat so we know channel count,
        # and (b) make sure Detect.stride is set (Ultralytics computes it from a probe pass
        # during model build, so it should already be set — this is just a safety check).
        with torch.no_grad():
            self.det_model.eval()
            self.det_model(torch.zeros(1, 3, 64, 64))   # tiny probe; layer 19 fires
        if self._kd_feat is None:
            raise RuntimeError(
                f"KD hook never fired — layer index {kd_feat_layer_idx} may be wrong."
            )
        self.feat_channels = int(self._kd_feat.shape[1])
        self._kd_feat = None  # clear; next real forward will refill

    # ----- head adjustment ---------------------------------------------------

    def _reinit_cls_head(self, new_nc: int):
        """Replace the final 1x1 Conv2d in each Detect.cv3 branch with a new one
        outputting `new_nc` channels. Backbone, DFL reg branch, and shared
        layers are untouched, so COCO pretraining still helps initial features.
        """
        det = self.det_model.model[-1]   # Detect module
        if det.nc == new_nc:
            return

        det.nc = new_nc
        det.no = new_nc + det.reg_max * 4

        for branch in det.cv3:
            last = branch[-1]
            if not isinstance(last, nn.Conv2d):
                raise RuntimeError(
                    f"Expected nn.Conv2d at cv3[-1], found {type(last).__name__}. "
                    f"Detect head layout changed in this Ultralytics version."
                )
            new_conv = nn.Conv2d(
                in_channels=last.in_channels,
                out_channels=new_nc,
                kernel_size=last.kernel_size,
                stride=last.stride,
                padding=last.padding,
                bias=True,
            )
            branch[-1] = new_conv

        # Re-init biases the way Ultralytics intends — uses `det.stride`,
        # which is already set on the Detect module from the original load.
        if hasattr(det, "bias_init") and hasattr(det, "stride") and det.stride is not None:
            det.bias_init()

    # ----- forward + feature hook -------------------------------------------

    def _register_kd_hook(self):
        def hook(_, __, output):
            self._kd_feat = output

        # det_model.model is a Sequential of layers; index by config.
        seq = self.det_model.model
        if not (0 <= self.kd_feat_layer_idx < len(seq)):
            raise ValueError(
                f"kd_feat_layer_idx={self.kd_feat_layer_idx} out of range "
                f"(len={len(seq)})."
            )
        seq[self.kd_feat_layer_idx].register_forward_hook(hook)

    def forward(self, x: torch.Tensor):
        return self.det_model(x)

    def forward_with_features(self, x: torch.Tensor):
        out = self.det_model(x)
        feat = self._kd_feat
        return out, feat

    # ----- save / load helpers ----------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.det_model.state_dict(*args, **kwargs)

    def load_state_dict(self, sd, strict: bool = True):
        return self.det_model.load_state_dict(sd, strict=strict)


def _reset_weights(m: nn.Module):
    """Fresh-init helper for `pretrained=False` mode."""
    for c in m.children():
        _reset_weights(c)
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


# ---------------------------------------------------------------------------
# Feature regressor (FitNets-style)
# ---------------------------------------------------------------------------


class FeatureRegressor(nn.Module):
    """1x1 conv (channel match) + bilinear resize (spatial match) -> teacher shape.

    Mirrors the tutorial's `regressor` layer but adds spatial alignment because
    the YOLOv11s feature map and the teacher feature map differ both in
    channel count and grid size.
    """

    def __init__(self, in_channels: int, out_channels: int, target_grid: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.target_grid = target_grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-1] != self.target_grid or x.shape[-2] != self.target_grid:
            x = F.interpolate(
                x, size=(self.target_grid, self.target_grid),
                mode="bilinear", align_corners=False,
            )
        return x
