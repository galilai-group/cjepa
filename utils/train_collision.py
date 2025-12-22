
import hydra
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from custom_models.collision_detector import build_collision_detector
from utils.dataset_slot_feature import SlotFeatureDataset
from pathlib import Path



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DINO_PATCH_SIZE = 14
# VIDEO_PATH_TRAIN = "/cs/data/people/hnam16/data/clevrer/videos/video_0[0-7]*.mp4"  # 00000-07999 for train
# VIDEO_PATH_VAL = "/cs/data/people/hnam16/data/clevrer/videos/video_0[89]*.mp4"  # 08000-09999 for val
# ANNOT_PATH_TRAIN = "/cs/data/people/hnam16/data/clevrer_annotation"
# ANNOT_PATH_VAL = "/cs/data/people/hnam16/data/clevrer_annotation"  # Use root, not subfolder
NUM_FRAMESKIP = 5
TRAIN_PATTERN="slot_features_0[0-7]*.npz"
VAL_PATTERN="slot_features_0[89]*.npz"




def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for slots, labels in tqdm(dataloader, desc="Training"):
        slots = slots.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        scores = model(slots)
        
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * slots.size(0)
        
        # Compute accuracy (threshold at 0.5 after sigmoid)
        preds = (torch.sigmoid(scores) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.numel()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for slots, labels in tqdm(dataloader, desc="Validating"):
            slots = slots.to(device)
            labels = labels.to(device)
            
            scores = model(slots)
            loss = criterion(scores, labels)
            
            total_loss += loss.item() * slots.size(0)
            
            preds = (torch.sigmoid(scores) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_samples
    
    # Calculate precision/recall
    all_preds = torch.cat(all_preds).flatten()
    all_labels = torch.cat(all_labels).flatten()
    
    true_positives = ((all_preds == 1) & (all_labels == 1)).sum().item()
    predicted_positives = (all_preds == 1).sum().item()
    actual_positives = (all_labels == 1).sum().item()
    
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return avg_loss, accuracy, precision, recall, f1


@hydra.main(version_base=None, config_path="../configs", config_name="config_test_causal")
def train_collision_detector(cfg):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Training hyperparameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WINDOW_SIZE = 8
    STRIDE = 2
    MODEL_TYPE = "conv"  # "conv" or "conv_diff"

    # Initialize wandb
    wandb.init(
        project="clevrer-collision-detection",
        config={
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "model_type": MODEL_TYPE,
            "pos_weight": 5.0,
        }
    )
    
    # Detect slot dimensions from a sample npz
    ckpt = cfg.checkpoint_path.split('/')[-1].split('.')[0]
    FEATURE_DIR = Path("/cs/data/people/hnam16/data/clevrer_feature") / ckpt


    npz_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith('.npz')])
    if not npz_files:
        raise RuntimeError(f"No slot feature .npz files found in {FEATURE_DIR}")
    sample = np.load(os.path.join(FEATURE_DIR, npz_files[0]))
    num_slots = int(sample['num_slots'])
    slot_dim = int(sample['slot_dim'])
    print(f"Detected num_slots={num_slots}, slot_dim={slot_dim}")

    # Build collision detector
    model = build_collision_detector(
        model_type=MODEL_TYPE,
        num_patches=num_slots,
        dim=slot_dim,
        hidden_dim=128
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Collision detector: {model}")

    # Prepare datasets (slot feature loading)
    print("Preparing datasets...")
    train_dataset = SlotFeatureDataset(FEATURE_DIR, window_size=WINDOW_SIZE, stride=STRIDE, pattern=TRAIN_PATTERN)
    val_dataset = SlotFeatureDataset(FEATURE_DIR, window_size=WINDOW_SIZE, stride=STRIDE, pattern=VAL_PATTERN)  # 분리 원하면 val용 디렉토리 따로 지정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Loss function with class weighting (collisions are rare)
    pos_weight = torch.tensor([5.0]).to(device)  # Upweight positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        # Validate
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        scheduler.step()
        # wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
        })
        #save current model
        save_path = f"collision_detector_{MODEL_TYPE}_latest_lr{LEARNING_RATE}_{ckpt}.pt"
        torch.save({
            'epoch': NUM_EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1': f1,
            'num_slots': num_slots,
            'slot_dim': slot_dim,
        }, save_path)
        print(f"Saved current model to {save_path}")
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            save_path = f"collision_detector_{MODEL_TYPE}_best_lr{LEARNING_RATE}_{ckpt}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'num_slots': num_slots,
                'slot_dim': slot_dim,
            }, save_path)
            print(f"Saved best model to {save_path}")
            wandb.run.summary["best_f1"] = best_f1
            wandb.run.summary["best_model_path"] = save_path
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train_collision_detector()
