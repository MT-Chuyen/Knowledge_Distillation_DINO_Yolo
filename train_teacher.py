"""Stage 1: fine-tune the DINOv2-L teacher with LoRA on the detection task.

Only the LoRA adapters and the detection head learn — DINOv2 backbone weights
are frozen by `DinoV2Teacher(freeze_backbone=True)`.

Run:
    python train_teacher.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import YoloAnimalDataset, build_targets, collate_fn
from evaluate import compute_map_50, predict
from losses import DetectionLoss
from models import DinoV2Teacher


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    model = DinoV2Teacher(
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

    n_trainable = count_trainable(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Teacher params trainable={n_trainable:,} / total={n_total:,}")

    criterion = DetectionLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr_teacher"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg["train"]["epochs_teacher"]
    )
    scaler = torch.amp.GradScaler(enabled=cfg["train"]["amp"])

    ckpt_path = Path(cfg["teacher"]["ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    for epoch in range(cfg["train"]["epochs_teacher"]):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"teacher epoch {epoch+1}/{cfg['train']['epochs_teacher']}")
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            obj_t, cls_t, box_t, pos = build_targets(
                targets, cfg["data"]["grid_size"], cfg["data"]["num_classes"], device
            )

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg["train"]["amp"]):
                out = model(imgs)
                loss, parts = criterion(out, obj_t, cls_t, box_t, pos)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            scaler.step(optim)
            scaler.update()

            running += float(loss.detach())
            n += 1
            pbar.set_postfix(
                loss=f"{running/n:.3f}",
                obj=f"{float(parts['obj']):.2f}",
                cls=f"{float(parts['cls']):.2f}",
                giou=f"{float(parts['giou']):.2f}",
            )

        sched.step()

        preds, gts = predict(
            model, val_loader, device,
            conf_threshold=cfg["train"]["conf_threshold"],
            nms_iou=cfg["train"]["nms_iou"],
            img_size=cfg["data"]["img_size"],
            num_classes=cfg["data"]["num_classes"],
            amp=cfg["train"]["amp"],
        )
        map50, _ = compute_map_50(preds, gts, num_classes=cfg["data"]["num_classes"])
        print(f"  -> val mAP@0.5 = {map50:.4f}  (best {max(best_map, map50):.4f})")

        if map50 > best_map:
            best_map = map50
            # Only save the (small) trainable state — adapters + head.
            torch.save(
                {"trainable": model.trainable_state_dict(),
                 "epoch": epoch, "map50": map50, "config": cfg},
                ckpt_path,
            )
            print(f"  -> saved best to {ckpt_path}")

    print(f"\nBest teacher mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
