"""
Unified Training Engine for Ink Detection (Vesuvius Challenge)
Author: Romain Frossard
Institution: DHLAB, EPFL

Description:
This script handles the training of a Multi-Layer Perceptron (MLP) segmentation head 
on NVlabs/RADIO embeddings. It dynamically supports both full-RAM preloading (optimal 
for lighter 512x512 datasets) and asynchronous chunked loading (mandatory for massive 
256x256 datasets) to maximize hardware efficiency without Out-Of-Memory (OOM) errors.

Usage Examples:
    # For 512x512 patches (Preload strategy)
    python lib/train.py --hdf5_path data/train_512.h5 --weights_dir weights/ --batch_size 32768 --strategy preload
    
    # For 256x256 patches (Async chunking strategy)
    python lib/train.py --hdf5_path data/train_256.h5 --weights_dir weights/ --batch_size 262144 --strategy async
"""

import os
import gc
import h5py
import torch
import random
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# --- 1. ARCHITECTURE & LOSS FUNCTIONS ---
# ==============================================================================

class RadioSegmentationHead(nn.Module):
    """
    A custom Multi-Layer Perceptron (MLP) designed to act as the segmentation 
    head for the frozen Vision Foundation Model (NVlabs/RADIO).
    
    Args:
        input_dim (int): Dimensionality of the incoming token embeddings. Defaults to 4096.
        hidden_dim (int): Dimensionality of the hidden layer. Defaults to 256.
        output_dim (int): Dimensionality of the output prediction. Defaults to 1.
    """
    def __init__(self, input_dim=4096, hidden_dim=256, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)


class DiceBCELoss(nn.Module):
    """
    A hybrid loss function combining Binary Cross-Entropy (BCE) and Dice Loss.
    Heavily penalizes false negatives on faint ink strokes via a massive positive weight.
    
    Args:
        pos_weight_val (float): The multiplier for the positive class (ink) in the BCE component.
    """
    def __init__(self, pos_weight_val=50.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight_val])
        
    def forward(self, inputs, targets, smooth=1e-5):
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight.to(inputs.device)
        )
        
        return bce_loss + (1.5 * (1.0 - dice_score))

# ==============================================================================
# --- 2. DATA LOADING STRATEGIES ---
# ==============================================================================

def preload_hdf5_to_ram(hdf5_path):
    """
    Loads the entire HDF5 dataset into system RAM. Recommended only for lighter 
    datasets (e.g., 512x512 extraction) where total tokens do not exceed available memory.
    """
    print(f"[*] Preloading entire HDF5 database into RAM from: {hdf5_path}")
    data_by_image = {}

    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
        images = list(set([key.split('_')[0] for key in keys]))
        
        for img in images:
            data_by_image[img] = {'features': [], 'labels': []}
            
        for key in tqdm(keys, desc="Loading data into RAM"):
            img_prefix = key.split('_')[0]
            data_by_image[img_prefix]['features'].append(torch.tensor(f[key]['features'][:]).half())
            data_by_image[img_prefix]['labels'].append(torch.tensor(f[key]['labels'][:]).half())

    for img_prefix in images:
        if data_by_image[img_prefix]['features']:
            data_by_image[img_prefix]['features'] = torch.cat(data_by_image[img_prefix]['features'], dim=0)
            data_by_image[img_prefix]['labels'] = torch.cat(data_by_image[img_prefix]['labels'], dim=0)

    return data_by_image, images

# ==============================================================================
# --- 3. OPTIMIZED TRAINING ENGINE ---
# ==============================================================================

def train_model(hdf5_path, weights_dir, epochs=50, batch_size=32768, strategy='async', keys_per_chunk=350, patience_limit=5, device='cuda'):
    """
    Executes the training loop, dynamically adapting the memory management strategy.
    """
    print(f"\n{'='*60}\nSTARTING MODEL TRAINING | STRATEGY: {strategy.upper()}\n{'='*60}")
    os.makedirs(weights_dir, exist_ok=True)

    if not os.path.exists(hdf5_path):
        print(f"[CRITICAL ERROR] HDF5 database not found at {hdf5_path}")
        return

    # Initialize Model & Optimization
    head = RadioSegmentationHead().to(device)
    
    if int(torch.__version__.split('.')[0]) >= 2:
        print("[*] PyTorch 2.x detected: Activating model compilation...")
        head = torch.compile(head)

    optimizer = optim.Adam(head.parameters(), lr=0.001)
    criterion = DiceBCELoss(pos_weight_val=50.0).to(device)
    scaler = torch.amp.GradScaler(device)
    head.train()

    best_loss = float('inf')
    patience_counter = 0

    # ---------------------------------------------------------
    # STRATEGY A: PRELOAD (Best for 512x512)
    # ---------------------------------------------------------
    if strategy == 'preload':
        preloaded_data, images = preload_hdf5_to_ram(hdf5_path)
        total_tokens = sum([preloaded_data[img]['features'].shape[0] for img in images])
        print(f"[*] Total tokens loaded: {total_tokens:,}\n")

        for epoch in range(epochs):
            epoch_loss = 0
            steps = 0
            random.shuffle(images)

            for img in tqdm(images, desc=f"Epoch {epoch+1:02d}/{epochs}"):
                feats = preloaded_data[img]['features']
                labs = preloaded_data[img]['labels']
                N = feats.shape[0]
                indices = torch.randperm(N)

                for i in range(0, N, batch_size):
                    batch_idx = indices[i:i + batch_size]
                    b_feats = feats[batch_idx].to(device, non_blocking=True).float()
                    b_labs = labs[batch_idx].to(device, non_blocking=True).float()

                    optimizer.zero_grad()
                    with torch.amp.autocast(device, dtype=torch.float16):
                        predictions = head(b_feats).view(-1)
                        loss = criterion(predictions, b_labs)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    steps += 1

            avg_loss = epoch_loss / steps
            print(f"[*] End of Epoch {epoch+1:02d} | Average Loss: {avg_loss:.5f}")
            
            # Save final production model behavior (V4 behavior)
            final_weights_path = os.path.join(weights_dir, 'radio_production_model.pth')
            torch.save(head.state_dict(), final_weights_path)

    # ---------------------------------------------------------
    # STRATEGY B: ASYNC CHUNKING (Best for 256x256)
    # ---------------------------------------------------------
    elif strategy == 'async':
        with h5py.File(hdf5_path, 'r') as f:
            all_keys = list(f.keys())
        
        def load_chunk(keys):
            f_list, l_list = [], []
            with h5py.File(hdf5_path, 'r') as f:
                for k in keys:
                    f_list.append(torch.from_numpy(f[k]['features'][:]))
                    l_list.append(torch.from_numpy(f[k]['labels'][:]))
            return torch.cat(f_list), torch.cat(l_list)

        for epoch in range(epochs):
            epoch_loss = 0
            steps = 0
            random.shuffle(all_keys)
            chunks = [all_keys[i:i + keys_per_chunk] for i in range(0, len(all_keys), keys_per_chunk)]
            pbar = tqdm(chunks, desc=f"Epoch {epoch+1:02d}/{epochs}")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_chunk, chunks[0])

                for i, _ in enumerate(pbar):
                    chunk_feats, chunk_labs = future.result()
                    if i + 1 < len(chunks):
                        future = executor.submit(load_chunk, chunks[i+1])

                    N = chunk_feats.shape[0]
                    indices = torch.randperm(N)

                    for j in range(0, N, batch_size):
                        batch_idx = indices[j:j + batch_size]
                        b_feats = chunk_feats[batch_idx].to(device, non_blocking=True).float()
                        b_labs = chunk_labs[batch_idx].to(device, non_blocking=True).float()

                        optimizer.zero_grad()
                        with torch.amp.autocast(device, dtype=torch.float16):
                            predictions = head(b_feats).view(-1)
                            loss = criterion(predictions, b_labs)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        epoch_loss += loss.item()
                        steps += 1

                    pbar.set_postfix({'loss': f"{epoch_loss/steps:.4f}"})
                    del chunk_feats, chunk_labs
                    gc.collect()

            avg_loss = epoch_loss / steps
            print(f"[*] End of Epoch {epoch+1:02d} | Average Loss: {avg_loss:.5f}")

            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(head.state_dict(), os.path.join(weights_dir, 'radio_model_best.pth'))
                print("    [+] New best loss! Model weights secured.")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"\n[!] EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                    break

    print("\n[*] Training fully completed.")

# ==============================================================================
# --- 4. MAIN EXECUTION PORTAL ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Training Loop for Vesuvius Ink Detection")
    parser.add_argument('--hdf5_path', type=str, required=True, help="Path to the HDF5 embedding database")
    parser.add_argument('--weights_dir', type=str, required=True, help="Directory to save the best model weights")
    parser.add_argument('--epochs', type=int, default=50, help="Maximum number of epochs")
    parser.add_argument('--batch_size', type=int, default=32768, help="Token batch size per forward pass")
    
    # The crucial strategy toggle
    parser.add_argument('--strategy', type=str, choices=['preload', 'async'], default='async', 
                        help="'preload' for 512x512 datasets (loads entirely into RAM), 'async' for 256x256 datasets (loads in chunks).")
    
    parser.add_argument('--keys_per_chunk', type=int, default=350, help="Number of chunks to load into RAM at once (async only)")
    parser.add_argument('--patience', type=int, default=5, help="Epochs to wait for improvement before early stopping (async only)")
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_model(
        hdf5_path=args.hdf5_path,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        strategy=args.strategy,
        keys_per_chunk=args.keys_per_chunk,
        patience_limit=args.patience,
        device=device
    )

if __name__ == "__main__":
    main()