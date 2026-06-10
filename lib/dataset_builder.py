"""
Dataset Builder for Ink Detection (Vesuvius Challenge)
Author: Romain Frossard
Institution: DHLAB, EPFL

Description:
This script processes raw macroscopic papyrus scans and their corresponding binary ink masks.
It applies Contrast Limited Adaptive Histogram Equalization (CLAHE), extracts 256x256 patches, 
and generates a compressed HDF5 database containing rich Vision Transformer (NVlabs/RADIO) 
embeddings. It employs a hard-negative mining strategy to handle severe class imbalance.

Usage:
    python lib/dataset_builder.py --img_dir data/images --mask_dir data/masks --out_h5 data/train.h5
"""

import os
import cv2
import h5py
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# --- 1. MACROSCOPIC PRE-PROCESSING & PATCHING ---

def apply_rgb_clahe(img_rgb, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an RGB image.
    
    To prevent color distortion, the image is first converted to the LAB color space. 
    CLAHE is strictly applied to the Lightness (L) channel before merging back to RGB.

    Args:
        img_rgb (numpy.ndarray): The input image array in RGB format.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid (tuple, optional): Size of grid for histogram equalization. Defaults to (8, 8).

    Returns:
        numpy.ndarray: The contrast-enhanced RGB image.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def process_full_image_to_patches(full_img_path, full_ink_mask_path, output_dirs, patch_size=256):
    """
    Processes a full high-resolution papyrus scan into smaller manageable patches.
    
    Calculates a physical gap mask (scanner background) directly on the raw image, 
    then applies CLAHE. It extracts spatial patches (e.g., 256x256) of the image, 
    the ink mask, and the gap mask, discarding patches that are >95% background.

    Args:
        full_img_path (str): Filepath to the raw macroscopic TIFF image.
        full_ink_mask_path (str): Filepath to the corresponding binary ink mask.
        output_dirs (dict): Dictionary containing target directories for 'images', 'ink_masks', and 'gap_masks'.
        patch_size (int, optional): The height and width of the extracted square patches. Defaults to 256.

    Returns:
        int: The total number of valid patches successfully extracted and saved.
    """
    filename = os.path.basename(full_img_path)
    base_name = os.path.splitext(filename)[0]

    img_raw = np.array(Image.open(full_img_path).convert('RGB'))
    ink_mask_full = np.array(Image.open(full_ink_mask_path).convert('L'))

    # Calculate threshold mask on the RAW image for scanner gaps
    gray_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    gap_mask_full = (gray_raw > 200)

    # Apply CLAHE to save the standardized patches
    img_clahe_full = apply_rgb_clahe(img_raw)

    h, w = img_clahe_full.shape[:2]
    patch_count = 0

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch_img = img_clahe_full[y:y + patch_size, x:x + patch_size]
            patch_ink = ink_mask_full[y:y + patch_size, x:x + patch_size]
            patch_gap = gap_mask_full[y:y + patch_size, x:x + patch_size]

            # Skip patches that are almost entirely scanner background
            if np.mean(patch_gap) > 0.95:
                continue

            p_name = f"{base_name}_patch_{patch_count:04d}.tif"
            Image.fromarray(patch_img).save(os.path.join(output_dirs['images'], p_name))
            Image.fromarray(patch_ink).save(os.path.join(output_dirs['ink_masks'], p_name))
            Image.fromarray((patch_gap * 255).astype(np.uint8)).save(os.path.join(output_dirs['gap_masks'], p_name))
            patch_count += 1

    return patch_count

def build_patches(img_dir, mask_dir, patch_out_dir, patch_size):
    """
    Orchestrates the generation of patches for all fragments in the input directory.
    
    Creates the necessary subdirectories and iterates over all supported image files, 
    passing them to the individual patching function.

    Args:
        img_dir (str): Directory containing all raw TIFF fragments.
        mask_dir (str): Directory containing all corresponding binary ink masks.
        patch_out_dir (str): Root directory where the output patches will be stored.
        patch_size (int): Size of the square patches to extract.

    Returns:
        dict: A dictionary containing the final paths to the three output subdirectories 
              ('images', 'ink_masks', 'gap_masks').
    """
    output_dirs = {
        'images': os.path.join(patch_out_dir, 'images'),
        'ink_masks': os.path.join(patch_out_dir, 'ink_masks'),
        'gap_masks': os.path.join(patch_out_dir, 'gap_masks')
    }
    
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    valid_extensions = ('.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg')
    allowed_images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    print(f"[*] Detected {len(allowed_images)} Fragments for patching.")
    total_patches = 0

    for img in tqdm(allowed_images, desc="Extracting Patches"):
        img_p = os.path.join(img_dir, img)
        msk_p = os.path.join(mask_dir, img)
        if os.path.exists(msk_p):
            total_patches += process_full_image_to_patches(img_p, msk_p, output_dirs, patch_size)
        else:
            print(f"[WARNING] Missing ink mask for {img}. Skipping.")

    print(f"[*] Patching complete. {total_patches} patches created in {patch_out_dir}")
    return output_dirs


# --- 2. DATASET & HDF5 EXTRACTION ---

class PapyrusPatchDataset(Dataset):
    """
    A PyTorch Dataset class designed to load and transform papyrus patches.
    
    Handles the loading of RGB patches, ink masks, and gap masks. It applies 
    high-resolution bicubic upscaling to the images (to 2048x2048) for the Vision 
    Transformer, and downscales the masks (to 128x128) to match the final feature map size.

    Args:
        patches_base_dir (str): The root directory containing 'images', 'ink_masks', and 'gap_masks'.
    """
    def __init__(self, patches_base_dir):
        self.img_dir = os.path.join(patches_base_dir, 'images')
        self.ink_dir = os.path.join(patches_base_dir, 'ink_masks')
        self.gap_dir = os.path.join(patches_base_dir, 'gap_masks')

        self.patch_names = [f for f in os.listdir(self.img_dir) if f.endswith('.tif')]

        self.transform_zoom = T.Compose([
            T.Resize((2048, 2048), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.mask_transform_zoom = T.Compose([
            T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        """
        Retrieves and transforms a single patch and its corresponding masks.
        
        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - img_tensor (torch.Tensor): The normalized RGB image tensor (3, 2048, 2048).
                - ink_tensor (torch.Tensor): The binary ink mask tensor (1, 128, 128).
                - gap_tensor (torch.Tensor): The boolean gap mask tensor (1, 128, 128).
                - patch_name (str): The original filename of the patch.
        """
        patch_name = self.patch_names[idx]
        img_raw = Image.open(os.path.join(self.img_dir, patch_name)).convert('RGB')
        ink_mask = Image.open(os.path.join(self.ink_dir, patch_name)).convert('L')
        gap_mask = Image.open(os.path.join(self.gap_dir, patch_name)).convert('L')

        img_tensor = self.transform_zoom(img_raw)

        ink_tensor = self.mask_transform_zoom(ink_mask)
        if ink_tensor.mean() > 0.5:
            ink_tensor = 1.0 - ink_tensor
        ink_tensor = (ink_tensor > 0.5).float()

        gap_tensor = self.mask_transform_zoom(gap_mask)
        gap_tensor = (gap_tensor > 0.5).bool()

        return img_tensor, ink_tensor, gap_tensor, patch_name

def extract_and_save_hdf5(patches_dir, hdf5_path, device='cuda'):
    """
    Extracts dense embeddings using NVlabs/RADIO and compiles them into an HDF5 database.
    
    This function utilizes a frozen Vision Foundation Model to process the papyrus patches 
    through multiple photometric augmentations. It applies Hard Negative Mining by 
    asymmetrically downsampling empty papyrus and scanner background tokens to handle 
    severe class imbalance before saving the resulting embeddings.

    Args:
        patches_dir (str): Directory containing the previously generated patches.
        hdf5_path (str): Destination filepath for the output .h5 database.
        device (str, optional): The computation device ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        None: The function writes directly to the specified HDF5 file.
    """
    print(f"[*] Starting feature extraction on device: {device.upper()}")

    dataset = PapyrusPatchDataset(patches_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    print("[*] Loading NVlabs/RADIO v2.5-l...")
    radio_v2 = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', skip_validation=True).to(device)
    radio_v2.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentations = {
        "orig": lambda i, m, g: (i, m, g),
        "brighter": lambda i, m, g: (TF.adjust_brightness(i, 1.2), m, g),
        "darker":   lambda i, m, g: (TF.adjust_brightness(i, 0.8), m, g),
        "contrast": lambda i, m, g: (TF.adjust_contrast(i, 1.2), m, g),
        "super_dark": lambda i, m, g: (TF.adjust_brightness(i, 0.5), m, g),
        "hyper_contrast": lambda i, m, g: (TF.adjust_contrast(i, 2.0), m, g)
    }

    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)

    with h5py.File(hdf5_path, 'w') as h5f:
        with torch.no_grad():
            for img_batch, ink_batch, gap_batch, patch_names in tqdm(dataloader, desc="Processing Batches"):
                img_batch = img_batch.to(device)
                ink_batch = ink_batch.to(device)
                gap_batch = gap_batch.to(device)

                for aug_name, aug_fn in augmentations.items():
                    aug_img, aug_ink, aug_gap = aug_fn(img_batch, ink_batch, gap_batch)
                    aug_img = normalize(aug_img)

                    summary, spatial = radio_v2(aug_img)

                    if len(spatial.shape) == 4:
                        spatial = spatial.permute(0, 2, 3, 1).reshape(spatial.shape[0], -1, spatial.shape[1])

                    for b_idx in range(img_batch.size(0)):
                        b_name = patch_names[b_idx].replace('.tif', '')

                        single_spatial = spatial[b_idx]
                        single_summary = summary[b_idx]

                        num_tokens = single_spatial.shape[0]
                        single_summary_expanded = single_summary.unsqueeze(0).expand(num_tokens, -1)
                        combined_tokens = torch.cat([single_spatial, single_summary_expanded], dim=-1)

                        flat_ink = aug_ink[b_idx].reshape(-1)
                        flat_gap = aug_gap[b_idx].reshape(-1)

                        is_ink = (flat_ink > 0)
                        mask_papyrus = (~flat_gap) & (~is_ink)

                        keep_papyrus_prob = torch.rand(mask_papyrus.shape, device=device)
                        keep_papyrus = mask_papyrus & (keep_papyrus_prob < 0.35)

                        keep_gap_prob = torch.rand(flat_gap.shape, device=device)
                        keep_gap = flat_gap & (keep_gap_prob < 0.15)

                        final_save_mask = is_ink | keep_papyrus | keep_gap

                        if final_save_mask.sum() == 0:
                            continue

                        valid_tokens = combined_tokens[final_save_mask].cpu().half().numpy()
                        valid_labels = flat_ink[final_save_mask].cpu().numpy()

                        group_name = f"{b_name}_{aug_name}"
                        group = h5f.create_group(group_name)
                        group.create_dataset('features', data=valid_tokens, compression="gzip", compression_opts=1)
                        group.create_dataset('labels', data=valid_labels, compression="gzip", compression_opts=1)

    print(f"[*] Extraction complete. Embeddings safely stored at {hdf5_path}")

def verify_hdf5_integrity(hdf5_path):
    """
    Performs a sanity check on the generated HDF5 database to detect corruption.
    
    Args:
        hdf5_path (str): Filepath to the HDF5 database to verify.

    Returns:
        None: Prints the validation status and the number of unique fragments found.
    """
    print(f"\n[*] Running sanity check on: {hdf5_path}")
    if not os.path.exists(hdf5_path):
        print("[CRITICAL ERROR] HDF5 file does not exist.")
        return

    try:
        with h5py.File(hdf5_path, 'r') as f:
            found_fragments = sorted(list(set([key.split('_')[0] for key in f.keys()])))
        
        print(f"[*] Detected {len(found_fragments)} unique base fragments in database.")
        print("[*] Database structure verified and healthy.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to read HDF5 file. Corrupt structure. Details: {e}")

# --- 3. MAIN EXECUTION PORTAL ---

def main():
    """
    Main entry point for the script. Handles argument parsing from the CLI 
    and orchestrates the data pipeline sequence.
    """
    parser = argparse.ArgumentParser(description="Build HDF5 Dataset for Vesuvius Ink Detection")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory containing raw TIFF images")
    parser.add_argument('--mask_dir', type=str, required=True, help="Directory containing binary TIFF ink masks")
    parser.add_argument('--patch_dir', type=str, default="data/intermediate_patches", help="Directory to store CLAHE patches")
    parser.add_argument('--hdf5_out', type=str, required=True, help="Output path for the .h5 embedding database")
    parser.add_argument('--patch_size', type=int, default=256, help="Window size for patch extraction")
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: Generate Patches
    build_patches(args.img_dir, args.mask_dir, args.patch_dir, args.patch_size)

    # Step 2: Extract Deep Features into HDF5
    extract_and_save_hdf5(args.patch_dir, args.hdf5_out, device=device)

    # Step 3: Validate output
    verify_hdf5_integrity(args.hdf5_out)

if __name__ == "__main__":
    main()