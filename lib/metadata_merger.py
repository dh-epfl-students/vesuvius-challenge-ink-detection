"""
Metadata Merger and Dataset Standardizer for Papyrology
Author: Romain Frossard
Institution: DHLAB, EPFL

Description:
This script consolidates disparate historical datasets (Duke University and 
University of Oslo Archives) into a unified, machine-learning-ready format. 
It assigns sequential surrogate IDs (e.g., 001.tif), converts all raw images 
to lossless LZW-compressed TIFFs, dynamically scrapes missing metadata from 
institutional web records, and exports a clean, standardized CSV manifest.

Usage:
    python lib/metadata_merger.py --duke_csv data/duke/metadata.csv \
                                  --duke_img data/duke/images \
                                  --oslo_meta data/oslo/metadata \
                                  --oslo_img data/oslo/images \
                                  --out_dir data/Unified_Dataset
"""

import os
import re
import csv
import json
import time
import argparse
import requests
import urllib3
from bs4 import BeautifulSoup
from PIL import Image

# Disable SSL warnings for external archival scraping
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# --- 1. UTILITY FUNCTIONS ---
# ==============================================================================

def convert_and_rename_image(source_path, dest_path):
    """
    Reads the source image, ensures loss-free RGB conversion, and saves it.

    Args:
        source_path (str): Filepath of the original raw image.
        dest_path (str): Filepath for the standardized output image.

    Returns:
        bool: True if conversion and saving were successful, False otherwise.
    """
    if not os.path.exists(source_path):
        return False
    try:
        with Image.open(source_path) as img:
            if img.mode in ('P', 'RGBA'):
                img = img.convert('RGB')
            # Save using LZW compression to optimize storage without data loss
            img.save(dest_path, format='TIFF', compression='tiff_lzw')
        return True
    except Exception as e:
        print(f"[IMAGE ERROR] Failed to process {source_path}: {e}")
        return False

def scrape_oslo_description(url):
    """
    Scrapes an OPES Oslo web page to extract physical and historical descriptions.
    
    Given the variable structure of the legacy web archive, it aggressively 
    extracts all definition lists, tables, and paragraphs, condensing them 
    into a single descriptive string.

    Args:
        url (str): URL pointing to the specific papyrus record.

    Returns:
        str: A consolidated string containing the extracted metadata.
    """
    try:
        resp = requests.get(url, verify=False, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            metadata_blocks = soup.find_all(['dl', 'table', 'p'])
            
            if metadata_blocks:
                raw_text = " | ".join([block.get_text(separator=' ', strip=True) for block in metadata_blocks])
                clean_text = re.sub(r'\s+', ' ', raw_text)
                clean_text = re.sub(r'(\|\s*)+', '| ', clean_text).strip(' |')
                
                if len(clean_text) > 800:
                    return clean_text[:800] + "... [See Record URL for full details]"
                return clean_text
    except Exception:
        pass
    
    return "Historical and physical description available at Record URL."

def clean_license(license_str):
    """
    Strips operational cross-reference tags to leave a clean legal license string.

    Args:
        license_str (str): The raw license string extracted from the JSON.

    Returns:
        str: The cleaned license designation.
    """
    if "(Verified" in license_str:
        return license_str.split("(Verified")[0].strip()
    return license_str

# ==============================================================================
# --- 2. MAIN COMPILATION ENGINE ---
# ==============================================================================

def build_ultimate_dataset(duke_csv, duke_img_dir, oslo_meta_dir, oslo_img_dir, out_dir):
    """
    Orchestrates the metadata harmonization and image standardization pipeline.

    Iterates through both Duke and Oslo datasets, renames files using a global 
    sequential ID, extracts features, and outputs a unified CSV and image folder.

    Args:
        duke_csv (str): Path to the raw Duke metadata CSV.
        duke_img_dir (str): Directory containing raw Duke TIFFs.
        oslo_meta_dir (str): Directory containing Oslo JSON metadata files.
        oslo_img_dir (str): Directory containing raw Oslo images.
        out_dir (str): Destination directory for the unified dataset.
    """
    out_img_dir = os.path.join(out_dir, "images")
    out_csv_path = os.path.join(out_dir, "metadata.csv")
    os.makedirs(out_img_dir, exist_ok=True)

    unified_data = []
    global_counter = 1

    # --- PROCESSING DUKE UNIVERSITY COHORT ---
    print("\n[INFO] Processing Duke University fragments...")
    try:
        with open(duke_csv, 'r', encoding='utf-8') as duke_file:
            reader = csv.DictReader(duke_file)
            for row in reader:
                if row.get('Error'):
                    continue
                
                global_id = f"{global_counter:03d}"
                inv = row.get('Inventory', 'Unknown')
                
                source_img = os.path.join(duke_img_dir, row.get('Filename', ''))
                dest_img = os.path.join(out_img_dir, f"{global_id}.tif")
                
                if convert_and_rename_image(source_img, dest_img):
                    desc_parts = []
                    for field in ['Title', 'Subject (Date)', 'Material', 'Note']:
                        val = row.get(field, '').replace('\n', ' ').strip()
                        if val:
                            desc_parts.append(val)
                    merged_description = " | ".join(desc_parts)

                    unified_row = {
                        "Global_ID": global_id,
                        "Institution": "Duke University",
                        "Inventory_Number": inv,
                        "Official_Citation": f"Papyrus P.Duk.inv. {inv}, David M. Rubenstein Rare Book & Manuscript Library, Duke University",
                        "Record_URL": row.get('URL', ''),
                        "License_Info": "Please refer to Duke University Rubenstein Library guidelines.",
                        "Description": merged_description
                    }
                    unified_data.append(unified_row)
                    global_counter += 1
    except Exception as e:
        print(f"[ERROR] Duke processing failed: {e}")

    # --- PROCESSING UNIVERSITY OF OSLO COHORT ---
    print("\n[INFO] Processing University of Oslo fragments...")
    try:
        json_files = [f for f in os.listdir(oslo_meta_dir) if f.endswith('.json')]
        for filename in json_files:
            filepath = os.path.join(oslo_meta_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                global_id = f"{global_counter:03d}"
                inv = data.get('original_image_name', '').split('.')[0]
                
                nom_sans_extension = os.path.splitext(data.get('original_image_name'))[0]
                original_img_name_tif = f"{data.get('fragment_id')}_{nom_sans_extension}.tif"
                
                source_img = os.path.join(oslo_img_dir, original_img_name_tif)
                dest_img = os.path.join(out_img_dir, f"{global_id}.tif")
                
                if convert_and_rename_image(source_img, dest_img):
                    url = data.get('record_url', '')
                    print(f"       -> Extracting web metadata for {inv} (ID: {global_id})...")
                    
                    oslo_description = scrape_oslo_description(url)
                    time.sleep(0.4) # Polite delay to avoid hammering the university server
                    
                    unified_row = {
                        "Global_ID": global_id,
                        "Institution": "University of Oslo",
                        "Inventory_Number": inv,
                        "Official_Citation": data.get('required_citation', ''),
                        "Record_URL": url,
                        "License_Info": clean_license(data.get('image_license', '')),
                        "Description": oslo_description
                    }
                    unified_data.append(unified_row)
                    global_counter += 1
    except Exception as e:
         print(f"[ERROR] Oslo processing failed: {e}")

    # --- EXPORTING FINAL DATASET ---
    print("\n[INFO] Compiling final dataset structure...")
    headers = [
        "Global_ID", "Institution", "Inventory_Number", "Official_Citation", 
        "Record_URL", "License_Info", "Description"
    ]

    with open(out_csv_path, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(unified_data)
        
    print(f"\n[SUCCESS] {len(unified_data)} fragments successfully harmonized.")
    print(f"[SUCCESS] Dataset ID range: 001 to {global_counter - 1:03d}.")
    print(f"[SUCCESS] Metadata manifest saved to: {out_csv_path}")

# ==============================================================================
# --- 3. CLI ENTRY POINT ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unify and standardize historical papyri datasets.")
    parser.add_argument('--duke_csv', type=str, required=True, help="Path to the raw Duke metadata CSV.")
    parser.add_argument('--duke_img', type=str, required=True, help="Directory containing raw Duke TIFFs.")
    parser.add_argument('--oslo_meta', type=str, required=True, help="Directory containing Oslo JSON metadata.")
    parser.add_argument('--oslo_img', type=str, required=True, help="Directory containing raw Oslo images.")
    parser.add_argument('--out_dir', type=str, required=True, help="Destination directory for the unified dataset.")
    
    args = parser.parse_args()

    build_ultimate_dataset(
        duke_csv=args.duke_csv,
        duke_img_dir=args.duke_img,
        oslo_meta_dir=args.oslo_meta,
        oslo_img_dir=args.oslo_img,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()