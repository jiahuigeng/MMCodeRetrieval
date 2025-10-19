#!/usr/bin/env python3
"""
Convert ChartQA dataset to multimodal retrieval test format.
Only generates test set with 4 fields: qry_text, qry_img_path, tgt_text, tgt_img_path
"""

import os
import json
import pandas as pd
from PIL import Image
import io
from pathlib import Path

def save_image_from_bytes(image_bytes, output_path):
    """Save image from bytes to file"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def convert_chartqa_to_test_format(parquet_path, output_dir):
    """
    Convert ChartQA dataset to test format.
    
    For ChartQA, we'll use:
    - qry_text: The question
    - qry_img_path: Path to the chart image
    - tgt_text: The answer (as a list)
    - tgt_img_path: Same as qry_img_path (as a list)
    """
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Read the parquet file
    print(f"Reading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples")
    
    test_samples = []
    
    for idx, row in df.iterrows():
        # Extract image bytes
        image_data = row['image']
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
        else:
            print(f"Warning: No image bytes found for row {idx}")
            continue
        
        # Save image
        image_filename = f"chartqa_{idx}.png"
        image_path = images_dir / image_filename
        relative_image_path = f"images/{image_filename}"
        
        if not save_image_from_bytes(image_bytes, image_path):
            print(f"Warning: Failed to save image for row {idx}")
            continue
        
        # Create test sample
        test_sample = {
            "qry_text": f"<|image_1|>{row['question']}",  # Add required prefix
            "qry_img_path": relative_image_path,
            "tgt_text": [str(row['answer'])],  # Convert to list of strings
            "tgt_img_path": [relative_image_path]  # Same image, as list
        }
        
        test_samples.append(test_sample)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} samples...")
    
    # Save test set
    test_file = output_dir / "test.jsonl"
    print(f"Saving {len(test_samples)} test samples to {test_file}...")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed!")
    print(f"Test set: {len(test_samples)} samples")
    print(f"Images saved to: {images_dir}")
    print(f"Test file: {test_file}")

def main():
    # Paths
    parquet_path = "/home/a/Projects/MMCodeRetrieval/datasets/ChartQA/data/test-00000-of-00001.parquet"
    output_dir = "/home/a/Projects/MMCodeRetrieval/MMCoIR/chartqa"
    
    # Check if input file exists
    if not os.path.exists(parquet_path):
        print(f"Error: Input file not found: {parquet_path}")
        return
    
    # Convert dataset
    convert_chartqa_to_test_format(parquet_path, output_dir)

if __name__ == "__main__":
    main()