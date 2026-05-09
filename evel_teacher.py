import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from data import YoloAnimalDataset, collate_fn
from evaluate import compute_map_50, predict
from models import DinoV2Teacher

def evaluate_on_test_set(config_path="config.yaml"):
    # 1. Setup
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the Test Dataset
    # We point directly to the /test folder
    test_ds = YoloAnimalDataset(
        root=cfg["data"]["root"], 
        split="test",  # This will look into /home/chuyenmt/code/full_animal/test
        img_size=cfg["data"]["img_size"], 
        num_classes=cfg["data"]["num_classes"],
        augment=False
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=cfg["data"]["batch_size"], 
        shuffle=False, num_workers=cfg["data"]["num_workers"], 
        collate_fn=collate_fn
    )
    print(f"Loaded test set with {len(test_ds)} images.")

    # 3. Initialize & Load Best Weights
    model = DinoV2Teacher(
        num_classes=cfg["data"]["num_classes"],
        grid_size=cfg["data"]["grid_size"],
        embed_dim=cfg["teacher"]["embed_dim"],
        head_hidden=cfg["teacher"]["head_hidden"],
        dinov2_name=cfg["teacher"]["dinov2_name"],
        lora_r=cfg["teacher"]["lora_r"],
        lora_alpha=cfg["teacher"]["lora_alpha"],
        lora_dropout=cfg["teacher"]["lora_dropout"],
        freeze_backbone=True
    ).to(device)

    ckpt = torch.load(cfg["teacher"]["ckpt"], map_location=device)
    model.load_trainable_state(ckpt["trainable"])
    model.eval()
    print(f"Successfully loaded checkpoint from epoch {ckpt['epoch']} (Best Val mAP: {ckpt['map50']:.4f})")

    # 4. Run Evaluation
    print("Running inference on test set...")
    preds, gts = predict(
        model, test_loader, device,
        conf_threshold=cfg["train"]["conf_threshold"],
        nms_iou=cfg["train"]["nms_iou"],
        amp=cfg["train"]["amp"]
    )
    
    # 5. Calculate Metrics
    map50, class_aps = compute_map_50(preds, gts, num_classes=cfg["data"]["num_classes"])
    
    print("\n" + "="*30)
    print(f"FINAL TEST mAP@0.5: {map50:.4f}")
    print("="*30)
    
    # Optional: Print per-class accuracy
    for i, ap in enumerate(class_aps):
        print(f"Class {i}: AP = {ap:.4f}")

if __name__ == "__main__":
    evaluate_on_test_set()