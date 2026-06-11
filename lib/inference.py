"""
Production Inference and Hybrid Post-Processing Pipeline
Author: Romain Frossard
Institution: DHLAB, EPFL

Description:
This script performs sliding window inference on full-resolution historical papyri
using a frozen NVlabs/RADIO Vision Foundation Model and a custom MLP segmentation head.
It aggregates token probabilities and applies a rigorous hybrid post-processing sequence 
(Edge Artifact Suppression, Semantic-Guided Contours, and Morphological Reconnection)
to generate crisp, publication-ready binary ink masks.

Usage:
    python lib/inference.py --img_dir data/raw --weights weights/radio_model.pth --out_dir results/
"""

import os
import math
import cv2
import torch
import argparse
import traceback
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg') # Ensures matplotlib runs safely on headless servers
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. ARCHITECTURE ---
# ==============================================================================

class RadioSegmentationHead(nn.Module):
    """
    Multi-Layer Perceptron (MLP) segmentation head.
    Must perfectly match the architecture used during the training phase.
    """
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

# ==============================================================================
# --- 2. HYBRID POST-PROCESSING ---
# ==============================================================================

def apply_hybrid_post_processing(raw_img, full_prob_grid, gap_mask_full):
    """
    Transforms the raw probability matrix into a clean binary mask according 
    to the specific three-step methodology outlined in the report.

    Args:
        raw_img (np.ndarray): The original full-resolution RGB image.
        full_prob_grid (np.ndarray): The averaged probability logits from the AI.
        gap_mask_full (np.ndarray): Boolean mask indicating scanner background.

    Returns:
        np.ndarray: The final binary ink mask (0 for background, 1 for ink).
    """
    original_h, original_w = raw_img.shape[:2]
    gray_original = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)

    # --- Pre-computation: Papyrus Silhouette & Core Probabilities ---
    blurred_for_mask = cv2.GaussianBlur(gray_original, (15, 15), 0)
    _, robust_papyrus_mask = cv2.threshold(blurred_for_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    full_prob_grid_cleaned = full_prob_grid.copy()
    
    # Extra Safety: 18px inner boundary erosion to prevent edge bleeding
    kernel_edge = np.ones((18, 18), np.uint8)
    papyrus_safe_zone = cv2.erode(robust_papyrus_mask, kernel_edge, iterations=1)
    full_prob_grid_cleaned[papyrus_safe_zone == 0] = 0.0

    # --- Step 2: Semantic-Guided Contours ---
    # Enhancing physical contrast via CLAHE before adaptive thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_original)
    gray_clahe_smooth = cv2.medianBlur(gray_clahe, 3) 
    
    # Adaptive threshold optimized at C=4 (per report)
    physical_edges = cv2.adaptiveThreshold(
        gray_clahe_smooth, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=41, C=4
    )
    
    kernel_edges = np.ones((3, 3), np.uint8)
    physical_edges = cv2.morphologyEx(physical_edges, cv2.MORPH_OPEN, kernel_edges)

    # --- Step 1: Edge Artifact Suppression ---
    # 7x7 Morphological dilation on background gaps to zero-out hallucinated tears
    kernel_tear = np.ones((7, 7), np.uint8)
    tear_edges = cv2.dilate(gap_mask_full.astype(np.uint8), kernel_tear, iterations=1)
    physical_edges[tear_edges == 1] = 0

    # Fusing Neural Stencil (P > 0.55) with structural edges via logical AND
    core_ai = (full_prob_grid_cleaned > 0.55).astype(np.uint8)
    fringe_ai = (full_prob_grid_cleaned > 0.25).astype(np.uint8) 

    growth_zone = cv2.bitwise_and(fringe_ai, physical_edges)
    combined_mask = cv2.bitwise_or(core_ai, growth_zone)
    
    # Retain only structural elements attached to high-confidence AI cores
    num_labels, labels = cv2.connectedComponents(combined_mask, connectivity=8)
    reconstructed_ink = np.zeros_like(combined_mask)
    for i in range(1, num_labels):
        if np.any(core_ai[labels == i]):
            reconstructed_ink[labels == i] = 1

    # --- Step 3: Morphological Reconnection ---
    # Localized 4x4 morphological closing to reconnect fractured characters
    kernel_micro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    final_ink = cv2.morphologyEx(reconstructed_ink, cv2.MORPH_CLOSE, kernel_micro)

    # Extra Safety: 30-pixel dust removal
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_ink, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            final_ink[labels == i] = 0

    # Final gap enforcement
    kernel_gap = np.ones((5, 5), np.uint8)
    gap_mask_eroded = cv2.erode(gap_mask_full.astype(np.uint8), kernel_gap, iterations=1)
    final_ink[gap_mask_eroded == 1] = 0

    return final_ink

# ==============================================================================
# --- 3. SLIDING WINDOW INFERENCE ENGINE ---
# ==============================================================================

def process_single_image(full_img_path, head, radio_v2, transform, args, device='cuda'):
    """Performs strided inference on a single macroscopic image."""
    base_name = os.path.basename(full_img_path)
    expected_mask = os.path.join(args.out_dir, "masks", base_name)
    
    if os.path.exists(expected_mask):
        return

    raw_img = np.array(Image.open(full_img_path).convert('RGB'))
    original_h, original_w = raw_img.shape[:2]
    gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)

    # Fast scanner background detection
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, white_bg = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    raw_silhouette = cv2.bitwise_not(white_bg)
    contours, _ = cv2.findContours(raw_silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    solid_papyrus = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(solid_papyrus, [contour], -1, 255, thickness=cv2.FILLED)

    outside_papyrus = cv2.bitwise_not(solid_papyrus)
    gap_mask_full = cv2.bitwise_or(outside_papyrus, white_bg)
    gap_mask_full = (gap_mask_full == 255)
    
    # Divisible Padding
    pad_h = math.ceil(original_h / args.patch_size) * args.patch_size - original_h
    pad_w = math.ceil(original_w / args.patch_size) * args.patch_size - original_w
    img_divisible = cv2.copyMakeBorder(raw_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_h, padded_w = img_divisible.shape[:2]

    full_prob_grid = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_grid = np.zeros((padded_h, padded_w), dtype=np.float32)

    stride = args.patch_size // 2 # 50% Spatial Overlap guaranteed
    batch_tensors, batch_coords = [], []

    def process_batch(tensors, coords):
        batch_input = torch.stack(tensors).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.float16):
                summary, spatial = radio_v2(batch_input)
                if len(spatial.shape) == 4:
                    spatial = spatial.permute(0, 2, 3, 1).reshape(spatial.shape[0], -1, spatial.shape[1])

                num_tokens = spatial.shape[1]
                summary_expanded = summary.unsqueeze(1).expand(-1, num_tokens, -1)
                combined_tokens = torch.cat([spatial, summary_expanded], dim=-1).view(-1, 4096)

                logits = head(combined_tokens)
                grid_size = int(math.sqrt(num_tokens))
                probs = torch.sigmoid(logits).view(-1, grid_size, grid_size).cpu().numpy().astype(np.float32)

        for i, (y, x) in enumerate(coords):
            probs_resized = cv2.resize(probs[i], (args.patch_size, args.patch_size), interpolation=cv2.INTER_CUBIC)
            full_prob_grid[y:y+args.patch_size, x:x+args.patch_size] += probs_resized
            count_grid[y:y+args.patch_size, x:x+args.patch_size] += 1.0

    # Strided Extraction
    for y in range(0, padded_h - args.patch_size + 1, stride):
        for x in range(0, padded_w - args.patch_size + 1, stride):
            patch_img = img_divisible[y:y+args.patch_size, x:x+args.patch_size]
            batch_tensors.append(transform(Image.fromarray(patch_img)))
            batch_coords.append((y, x))
            
            if len(batch_tensors) == args.batch_size:
                process_batch(batch_tensors, batch_coords)
                batch_tensors, batch_coords = [], []
                
    if batch_tensors:
        process_batch(batch_tensors, batch_coords)

    # Averaging overlapping predictions
    full_prob_grid = full_prob_grid / np.maximum(count_grid, 1.0)
    full_prob_grid = full_prob_grid[:original_h, :original_w]

    # --- Apply Targeted Post-Processing ---
    final_ink = apply_hybrid_post_processing(raw_img[:original_h, :original_w], full_prob_grid, gap_mask_full[:original_h, :original_w])

    # --- Export Results ---
    final_mask_img = ((1 - final_ink) * 255).astype(np.uint8) # Inverted for visibility
    cv2.imwrite(expected_mask, final_mask_img)

    # Generation of the Quality Control Overlay
    overlay_img = raw_img.copy()
    alpha = 0.4
    ink_pixels = (final_ink == 1)
    overlay_img[ink_pixels] = (overlay_img[ink_pixels] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(raw_img); axes[0].set_title("1. Original Papyrus", fontsize=18); axes[0].axis('off')
    axes[1].imshow(full_prob_grid, cmap='inferno', vmin=0, vmax=1); axes[1].set_title("2. AI Probability (Heatmap)", fontsize=18); axes[1].axis('off')
    axes[2].imshow(overlay_img); axes[2].set_title("3. Ink Overlay", fontsize=18); axes[2].axis('off')
    axes[3].imshow(final_mask_img, cmap='gray'); axes[3].set_title("4. Final Binary Mask", fontsize=18); axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "overlays", f"panel_{base_name}"), dpi=150) 
    fig.clf(); plt.close(fig) 

# ==============================================================================
# --- 4. MULTIPROCESSING WORKER ---
# ==============================================================================

def process_images_on_gpu(gpu_id, image_subset, args):
    """Initializes models on a specific GPU and processes a subset of images."""
    try:
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        
        radio_v2 = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', skip_validation=True).to(device)
        radio_v2.eval()

        head = RadioSegmentationHead(input_dim=4096).to(device)
        state_dict = torch.load(args.weights, map_location=device, weights_only=True)
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        head.load_state_dict(clean_state_dict)
        head.eval()

        transform = T.Compose([
            T.Resize((2048, 2048), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for img_name in tqdm(image_subset, desc=f"GPU {gpu_id} Worker", position=gpu_id):
            img_path = os.path.join(args.img_dir, img_name)
            process_single_image(img_path, head, radio_v2, transform, args, device=device)
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[CRITICAL ERROR] GPU {gpu_id} encountered a fatal error: {e}")
        traceback.print_exc()

# ==============================================================================
# --- 5. MAIN EXECUTION PORTAL ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sliding Window Inference and Post-Processing")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory containing raw full-resolution TIFFs")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained PyTorch .pth weights")
    parser.add_argument('--out_dir', type=str, required=True, help="Root directory for outputs (creates /masks and /overlays)")
    parser.add_argument('--patch_size', type=int, default=256, help="Spatial dimensions of the processing window")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for model inference")
    
    args = parser.parse_args()

    # Ensure output structure exists
    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "overlays"), exist_ok=True)

    print("[INFO] Pre-loading NVIDIA architecture to secure PyTorch cache...")
    _ = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', skip_validation=True)
    print("[INFO] Cache secured. Initiating inference protocols.")

    valid_exts = ('.tif', '.png', '.jpg')
    all_image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_exts)])
    
    print(f"\n[INFO] Starting production run on {len(all_image_files)} fragments.")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"[INFO] Multi-GPU infrastructure detected ({num_gpus} GPUs). Spawning parallel workers.")
        mp.set_start_method('spawn', force=True)
        
        # Simple split for 2 GPUs
        mid_point = len(all_image_files) // 2
        images_gpu0 = all_image_files[:mid_point]
        images_gpu1 = all_image_files[mid_point:]
        
        p1 = mp.Process(target=process_images_on_gpu, args=(0, images_gpu0, args))
        p2 = mp.Process(target=process_images_on_gpu, args=(1, images_gpu1, args))
        
        p1.start(); p2.start()
        p1.join(); p2.join()
    else:
        print("[INFO] Single GPU (or CPU) detected. Running sequential execution.")
        process_images_on_gpu(0, all_image_files, args)

    print("\n[SUCCESS] Inference completed. Quality Control overlays and Binary Masks successfully generated.")

if __name__ == "__main__":
    main()