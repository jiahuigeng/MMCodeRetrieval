#!/usr/bin/env python3
"""
Convert SVGenius dataset to multimodal retrieval format.
SVGenius contains SVG code and filenames, we'll create a test set where:
- qry_text: filename description with required prefix
- qry_img_path: generated SVG image file
- tgt_text: SVG code
- tgt_img_path: same as qry_img_path (since we only have one image per sample)
"""

import pandas as pd
import json
import os
from pathlib import Path
import cairosvg
from PIL import Image
import io

def create_output_dirs(output_dir):
    """Create output directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "images")).mkdir(parents=True, exist_ok=True)

def svg_to_png(svg_code, output_path, width=512, height=512):
    """Convert SVG code to PNG image."""
    try:
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_code.encode('utf-8'),
            output_width=width,
            output_height=height
        )
        
        # Save PNG file
        with open(output_path, 'wb') as f:
            f.write(png_data)
        
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False

def process_svgenius_data(input_files, output_dir, is_train=False):
    """Process SVGenius data and convert to multimodal retrieval format."""
    
    create_output_dirs(output_dir)
    
    all_data = []
    image_counter = 0
    
    # For training set, we need to collect all data first to generate negative samples
    if is_train:
        all_samples = []
        # First pass: collect all samples
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping...")
                continue
                
            print(f"Collecting samples from {input_file}...")
            df = pd.read_parquet(input_file)
            
            for idx, row in df.iterrows():
                image_filename = f"svgenius_{image_counter}.png"
                image_path = os.path.join(output_dir, "images", image_filename)
                
                if svg_to_png(row['svg_code'], image_path):
                    all_samples.append({
                        'svg_code': row['svg_code'],
                        'image_filename': image_filename,
                        'image_counter': image_counter
                    })
                    image_counter += 1
        
        # Second pass: create training entries with negative samples
        print("Generating training data with negative samples...")
        for i, sample in enumerate(all_samples):
            # Select a random negative sample (different from current)
            neg_idx = (i + len(all_samples) // 2) % len(all_samples)
            neg_sample = all_samples[neg_idx]
            
            entry = {
                "qry": "<|image_1|>\nYou are a skilled SVG designer. Given an input image, generate the corresponding SVG code that visually matches it. Only output valid SVG code enclosed in <svg>...</svg>. Use only <path> elements with \"fill\" and \"d\" attributes. Do not include explanations or comments.",
                "qry_image_path": f"images/{sample['image_filename']}",
                "pos_text": sample['svg_code'],
                "pos_image_path": f"images/{sample['image_filename']}",
                "neg_text": neg_sample['svg_code'],
                "neg_image_path": f"images/{neg_sample['image_filename']}"
            }
            
            all_data.append(entry)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} training samples...")
    
    else:
        # Test set processing
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping...")
                continue
                
            print(f"Processing {input_file}...")
            df = pd.read_parquet(input_file)
            
            for idx, row in df.iterrows():
                # Generate image filename
                image_filename = f"svgenius_{image_counter}.png"
                image_path = os.path.join(output_dir, "images", image_filename)
                
                # Convert SVG to PNG
                if svg_to_png(row['svg_code'], image_path):
                    # Test set format (4 keys as per programming_goals.md)
                    entry = {
                        "qry_text": "<|image_1|>\nYou are a skilled SVG designer. Given an input image, generate the corresponding SVG code that visually matches it. Only output valid SVG code enclosed in <svg>...</svg>. Use only <path> elements with \"fill\" and \"d\" attributes. Do not include explanations or comments.",
                        "qry_img_path": f"images/{image_filename}",
                        "tgt_text": [row['svg_code']],  # List format for test set
                        "tgt_img_path": [f"images/{image_filename}"]  # List format for test set
                    }
                    
                    all_data.append(entry)
                    image_counter += 1
                    
                    if image_counter % 50 == 0:
                        print(f"Processed {image_counter} samples...")
                else:
                    print(f"Failed to convert SVG for sample {idx} in {input_file}")
    
    return all_data

def main():
    # Training set files
    train_files = [
        "datasets/SVGenius/data/train-00000-of-00001.parquet"
    ]
    
    # Test set files
    test_files = [
        "datasets/SVGenius/data/easy-00000-of-00001.parquet",
        "datasets/SVGenius/data/medium-00000-of-00001.parquet", 
        "datasets/SVGenius/data/hard-00000-of-00001.parquet"
    ]
    
    # Output directory
    output_dir = "MMCoIR/SVGenius"
    
    print("Starting SVGenius dataset conversion...")
    print(f"Output directory: {output_dir}")
    
    # Process training data
    print("Processing training data...")
    train_data = process_svgenius_data(train_files, output_dir, is_train=True)
    
    # Save training set
    train_jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Process test data
    print("Processing test data...")
    test_data = process_svgenius_data(test_files, output_dir, is_train=False)
    
    # Save test set
    test_jsonl_path = os.path.join(output_dir, "test.jsonl")
    with open(test_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed!")
    print(f"Training set: {len(train_data)} samples saved to {train_jsonl_path}")
    print(f"Test set: {len(test_data)} samples saved to {test_jsonl_path}")
    print(f"Images saved to {os.path.join(output_dir, 'images')}")

if __name__ == "__main__":
    main()