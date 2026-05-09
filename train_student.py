"""Stage 2: train YOLOv11s student.

Two modes (mirroring the PyTorch tutorial's experiments):

  --mode baseline   YOLOv11s trained with the standard Ultralytics
                    `v8DetectionLoss` (box + cls + dfl) and no teacher.
                    This is the analogue of the tutorial's "lightweight network
                    trained from scratch with cross-entropy" baseline.

  --mode kd         Same supervised loss, plus two KD signals from the frozen
                    DINOv2-L teacher:
                      1.  Feature MSE through a 1x1 regressor (FitNets-style;
                          tutorial mechanism #3 / "RegressorMSE").
                      2.  CosineEmbeddingLoss on globally-pooled features
                          (tutorial mechanism #2 / "CosineLoss").
                    Logit KD is dropped — see README for why.

Run:
    python train_student.py --mode baseline
    python train_student.py --mode kd  --teacher-ckpt runs/teacher/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import YoloAnimalDataset, collate_fn
from evaluate import compute_map_50, predict
from losses import YoloLossAdapter, cosine_kd_loss, feature_kd_loss
from models import DinoV2Teacher, FeatureRegressor, YoloV11sStudent


def build_teacher(cfg, device, ckpt_path: str) -> DinoV2Teacher:
    teacher = DinoV2Teacher(
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
    sd = torch.load(ckpt_path, map_location=device)
    teacher.load_trainable_state_dict(sd["trainable"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["baseline", "kd"], required=True)
    parser.add_argument("--teacher-ckpt", default=None,
                        help="Required when --mode kd. Defaults to teacher.ckpt from config.")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | mode={args.mode}")

    train_ds = YoloAnimalDataset(
        root=cfg["data"]["root"], split="train",
        img_size=cfg["data"]["img_size"], num_classes=cfg["data"]["num_classes"],
        augment=True,
    )
    val_ds = YoloAnimalDataset(
        root=cfg["data"]["root"], split="val",
        img_size=cfg["data"]["img_size"], num_classes=cfg["data"]["num_classes"],
        augment=False,
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
        pin_memory=True,
    )

    student = YoloV11sStudent(
        num_classes=cfg["data"]["num_classes"],
        weights=cfg["student"]["yolo_weights"],
        kd_feat_layer_idx=cfg["student"]["kd_feat_layer_idx"],
        loss_box=cfg["yolo_loss"]["box"],
        loss_cls=cfg["yolo_loss"]["cls"],
        loss_dfl=cfg["yolo_loss"]["dfl"],
        pretrained=cfg["student"]["pretrained"],
    ).to(device)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"Student YOLOv11s params: {n_total:,}  | KD feat channels={student.feat_channels}")

    sup_loss = YoloLossAdapter(student.det_model)

    if args.mode == "kd":
        teacher_ckpt = args.teacher_ckpt or cfg["teacher"]["ckpt"]
        print(f"Loading teacher checkpoint: {teacher_ckpt}")
        teacher = build_teacher(cfg, device, teacher_ckpt)
        regressor = FeatureRegressor(
            in_channels=student.feat_channels,
            out_channels=cfg["teacher"]["embed_dim"],
            target_grid=cfg["data"]["grid_size"],
        ).to(device)
        ckpt_path = Path(cfg["student"]["ckpt_kd"])
    else:
        teacher, regressor = None, None
        ckpt_path = Path(cfg["student"]["ckpt_baseline"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    params = list(student.parameters())
    if regressor is not None:
        params += list(regressor.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr_student"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg["train"]["epochs_student"]
    )
    scaler = torch.amp.GradScaler(enabled=cfg["train"]["amp"])

    w = cfg["kd"]
    best_map = -1.0

    for epoch in range(cfg["train"]["epochs_student"]):
        student.train()
        if regressor is not None:
            regressor.train()
        running = {"total": 0.0, "sup": 0.0, "feat": 0.0, "cos": 0.0}
        n = 0
        pbar = tqdm(train_loader, desc=f"{args.mode} epoch {epoch+1}/{cfg['train']['epochs_student']}")
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg["train"]["amp"]):
                # forward_with_features: student-mode forward returns the list of
                # raw per-scale feature maps that v8DetectionLoss expects.
                preds, s_feat = student.forward_with_features(imgs)
                sup, sup_items = sup_loss(preds, targets, device)
                total = w["w_sup"] * sup

                if teacher is not None:
                    with torch.no_grad():
                        _, t_feat = teacher.forward_with_features(imgs)
                    s_feat_proj = regressor(s_feat)            # match teacher shape
                    feat_l = feature_kd_loss(s_feat_proj, t_feat)
                    cos_l = cosine_kd_loss(s_feat_proj, t_feat)
                    total = total + w["w_feat"] * feat_l + w["w_cosine"] * cos_l
                else:
                    feat_l = torch.tensor(0.0, device=device)
                    cos_l = torch.tensor(0.0, device=device)

            scaler.scale(total).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optim)
            scaler.update()

            n += 1
            running["total"] += float(total.detach())
            running["sup"] += float(sup.detach())
            running["feat"] += float(feat_l.detach())
            running["cos"] += float(cos_l.detach())
            pbar.set_postfix(
                tot=f"{running['total']/n:.3f}",
                sup=f"{running['sup']/n:.3f}",
                feat=f"{running['feat']/n:.3f}",
                cos=f"{running['cos']/n:.3f}",
            )

        sched.step()

        preds_all, gts_all = predict(
            student, val_loader, device,
            conf_threshold=cfg["train"]["conf_threshold"],
            nms_iou=cfg["train"]["nms_iou"],
            img_size=cfg["data"]["img_size"],
            num_classes=cfg["data"]["num_classes"],
            amp=cfg["train"]["amp"],
        )
        map50, _ = compute_map_50(preds_all, gts_all, num_classes=cfg["data"]["num_classes"])
        print(f"  -> val mAP@0.5 = {map50:.4f}  (best {max(best_map, map50):.4f})")

        if map50 > best_map:
            best_map = map50
            payload = {
                "model": student.state_dict(),
                "epoch": epoch, "map50": map50, "mode": args.mode, "config": cfg,
            }
            if regressor is not None:
                payload["regressor"] = regressor.state_dict()
            torch.save(payload, ckpt_path)
            print(f"  -> saved best to {ckpt_path}")

    print(f"\nBest student ({args.mode}) mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
