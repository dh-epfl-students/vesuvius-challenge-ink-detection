import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import random
import argparse

# --- 1. DATA LOADING & PARSING ---
def get_image_prefixes(hdf5_path):
    """Dynamically extracts unique image prefixes from HDF5 keys."""
    with h5py.File(hdf5_path, 'r') as f:
        # Assumes keys are formatted like "11_patch_0_0" or "papyrus_patch_1"
        prefixes = sorted(list(set([key.split('_')[0] for key in f.keys()])))
    print(f"Detected {len(prefixes)} unique images for LOOCV: {prefixes}")
    return prefixes

def preload_hdf5_to_ram(hdf5_path, allowed_images):
    """Loads the entire HDF5 file into RAM once. Tensors are kept in float16."""
    print(f"Opening HDF5 file from: {hdf5_path}")
    data_by_image = {img: {'features': [], 'labels': []} for img in allowed_images}

    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
        for key in tqdm(keys, desc="Loading data into RAM"):
            # Match the key with the correct image prefix
            prefix = key.split('_')[0]
            if prefix in allowed_images:
                data_by_image[prefix]['features'].append(torch.tensor(f[key]['features'][:]).half())
                data_by_image[prefix]['labels'].append(torch.tensor(f[key]['labels'][:]).half())

    # Concatenate lists into giant tensors
    for img_prefix in allowed_images:
        if data_by_image[img_prefix]['features']:
            data_by_image[img_prefix]['features'] = torch.cat(data_by_image[img_prefix]['features'], dim=0)
            data_by_image[img_prefix]['labels'] = torch.cat(data_by_image[img_prefix]['labels'], dim=0)

    return data_by_image

# --- 2. ARCHITECTURE & LOSS ---
class RadioSegmentationHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256, output_dim=1):
        super(RadioSegmentationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight_val=None):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight_val]) if pos_weight_val else None

    def forward(self, inputs, targets, smooth=1e-5):
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1.0 - dice_score

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight.to(inputs.device) if self.pos_weight is not None else None
        )
        return bce_loss + (1.5 * dice_loss)

# --- 3. METRICS ---
def calculate_metrics(predictions, targets, threshold=0.5):
    preds_binary = (torch.sigmoid(predictions) > threshold).cpu().numpy().astype(int)
    targets_binary = targets.cpu().numpy().astype(int)

    iou = jaccard_score(targets_binary, preds_binary, zero_division=0)
    dice = f1_score(targets_binary, preds_binary, zero_division=0)
    precision = precision_score(targets_binary, preds_binary, zero_division=0)
    recall = recall_score(targets_binary, preds_binary, zero_division=0)

    return iou, dice, precision, recall

# --- 4. BLAZING FAST LOOCV LOOP ---
def run_loocv(hdf5_path, weights_dir, epochs=50, batch_size=32768, pos_weight=50.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
    
    os.makedirs(weights_dir, exist_ok=True)

    # Dynamically fetch image names from the dataset
    images = get_image_prefixes(hdf5_path)
    
    if not images:
        raise ValueError("No images found in the HDF5 file. Check your data path or file structure.")

    # Load to CPU RAM
    preloaded_data = preload_hdf5_to_ram(hdf5_path, images)
    all_metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': []}

    for val_img in images:
        print(f"\n{'='*50}\nStarting LOOCV Fold: Validation on [{val_img}]\n{'='*50}")
        train_imgs = [img for img in images if img != val_img]

        head = RadioSegmentationHead(input_dim=4096).to(device)
        optimizer = optim.Adam(head.parameters(), lr=0.001) 
        criterion = DiceBCELoss(pos_weight_val=pos_weight).to(device)

        # --- TRAINING PHASE ---
        for epoch in range(epochs):
            head.train()
            epoch_loss = 0
            steps = 0

            random.shuffle(train_imgs) # Shuffle image order per epoch

            for img in train_imgs:
                feats = preloaded_data[img]['features']
                labs = preloaded_data[img]['labels']
                N = feats.shape[0]

                indices = torch.randperm(N)

                # Direct Tensor Slicing (Bypasses CPU DataLoader bottleneck)
                for i in range(0, N, batch_size):
                    batch_idx = indices[i:i + batch_size]

                    # Transfer slice to GPU, cast to float32
                    b_feats = feats[batch_idx].to(device, non_blocking=True).float()
                    b_labs = labs[batch_idx].to(device, non_blocking=True).float()

                    optimizer.zero_grad()
                    predictions = head(b_feats).view(-1)
                    loss = criterion(predictions, b_labs)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    steps += 1

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss/steps:.4f}")

        # --- VALIDATION PHASE ---
        head.eval()
        all_preds = []

        print(f"Evaluating fold on [{val_img}]...")
        with torch.no_grad():
            val_feats = preloaded_data[val_img]['features']
            val_labs = preloaded_data[val_img]['labels'].float() 
            N_val = val_feats.shape[0]

            for i in range(0, N_val, batch_size):
                b_feats = val_feats[i:i + batch_size].to(device, non_blocking=True).float()
                preds = head(b_feats).view(-1).cpu() 
                all_preds.append(preds)

        fold_preds = torch.cat(all_preds)

        iou, dice, precision, recall = calculate_metrics(fold_preds, val_labs)
        print(f"\nResults for [{val_img}] -> IoU: {iou:.4f} | Dice: {dice:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        all_metrics['IoU'].append(iou)
        all_metrics['Dice'].append(dice)
        all_metrics['Precision'].append(precision)
        all_metrics['Recall'].append(recall)

        torch.save(head.state_dict(), os.path.join(weights_dir, f'radio_head_fold_{val_img}.pth'))

    print(f"\n{'='*50}\nFINAL LOOCV AVERAGED METRICS (N={len(images)})\n{'='*50}")
    print(f"Mean IoU:       {np.mean(all_metrics['IoU']):.4f} ± {np.std(all_metrics['IoU']):.4f}")
    print(f"Mean Dice:      {np.mean(all_metrics['Dice']):.4f} ± {np.std(all_metrics['Dice']):.4f}")
    print(f"Mean Precision: {np.mean(all_metrics['Precision']):.4f} ± {np.std(all_metrics['Precision']):.4f}")
    print(f"Mean Recall:    {np.mean(all_metrics['Recall']):.4f} ± {np.std(all_metrics['Recall']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-One-Out Cross-Validation (LOOCV) for Vesuvius Ink Detection")
    parser.add_argument('--hdf5_path', type=str, required=True, help="Path to the validation HDF5 database (e.g., DIBCO data)")
    parser.add_argument('--weights_dir', type=str, required=True, help="Directory to save the fold weights")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs per fold")
    parser.add_argument('--batch_size', type=int, default=32768, help="Token batch size per forward pass")
    parser.add_argument('--pos_weight', type=float, default=50.0, help="Positive weight multiplier for BCE Loss")
    
    args = parser.parse_args()

    run_loocv(
        hdf5_path=args.hdf5_path,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight
    )