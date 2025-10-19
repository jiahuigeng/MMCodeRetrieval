#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader for rootsautomation/RICO-Screen2Words

This script downloads the RICO-Screen2Words dataset from HuggingFace Hub.
The dataset contains mobile app screenshots paired with natural language descriptions.

Usage:
    python download_rico_dataset.py [--cache_dir /path/to/cache] [--split train] [--streaming]

Example:
    python download_rico_dataset.py --cache_dir ./datasets --split train
    python download_rico_dataset.py --streaming  # For large datasets
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import HfApi
except ImportError as e:
    print(f"Error: Missing required packages. Please install them:")
    print("pip install datasets huggingface-hub")
    sys.exit(1)

def setup_hf_mirror(use_mirror=True, mirror_url="https://hf-mirror.com"):
    """
    Setup HuggingFace mirror configuration
    
    Args:
        use_mirror (bool): Whether to use mirror
        mirror_url (str): Mirror URL to use
    """
    if use_mirror:
        os.environ['HF_ENDPOINT'] = mirror_url
        print(f"Using HuggingFace mirror: {mirror_url}")
    else:
        # Remove mirror setting if it exists
        if 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        print("Using official HuggingFace Hub")

def check_dataset_info(dataset_name):
    """
    Check if the dataset exists and get basic information
    """
    try:
        api = HfApi()
        dataset_info = api.dataset_info(dataset_name)
        print(f"Dataset: {dataset_name}")
        print(f"Description: {dataset_info.description or 'No description available'}")
        print(f"Tags: {', '.join(dataset_info.tags) if dataset_info.tags else 'No tags'}")
        print(f"Downloads: {dataset_info.downloads or 'Unknown'}")
        return True
    except Exception as e:
        print(f"Error checking dataset info: {e}")
        return False

def download_dataset(dataset_name, cache_dir=None, split=None, streaming=False, trust_remote_code=False):
    """
    Download the specified dataset
    
    Args:
        dataset_name (str): Name of the dataset to download
        cache_dir (str): Directory to cache the dataset
        split (str): Specific split to download (train, test, validation)
        streaming (bool): Whether to use streaming mode
        trust_remote_code (bool): Whether to trust remote code execution
    
    Returns:
        dataset: The loaded dataset object
    """
    try:
        print(f"\nDownloading dataset: {dataset_name}")
        
        # Set up parameters
        kwargs = {
            'streaming': streaming,
            'trust_remote_code': trust_remote_code
        }
        
        if cache_dir:
            kwargs['cache_dir'] = cache_dir
            print(f"Cache directory: {cache_dir}")
        
        if split:
            kwargs['split'] = split
            print(f"Split: {split}")
        
        if streaming:
            print("Using streaming mode (no local download)")
        else:
            print("Downloading to local cache...")
        
        # Load the dataset
        dataset = load_dataset(dataset_name, **kwargs)
        
        if streaming:
            print("Dataset loaded in streaming mode")
        else:
            print("Dataset downloaded successfully!")
            
            # Print dataset information
            if hasattr(dataset, 'num_rows'):
                print(f"Number of rows: {dataset.num_rows:,}")
            elif isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    print(f"Split '{split_name}': {len(split_data):,} rows")
            
            # Print column information
            if hasattr(dataset, 'column_names'):
                print(f"Columns: {', '.join(dataset.column_names)}")
            elif isinstance(dataset, dict) and dataset:
                first_split = next(iter(dataset.values()))
                if hasattr(first_split, 'column_names'):
                    print(f"Columns: {', '.join(first_split.column_names)}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify the dataset name is correct")
        print("3. Try using --streaming flag for large datasets")
        print("4. Check if you need authentication for private datasets")
        return None

def show_sample_data(dataset, num_samples=3):
    """
    Display sample data from the dataset
    """
    try:
        print(f"\n--- Sample Data (first {num_samples} examples) ---")
        
        if hasattr(dataset, '__iter__'):
            # For streaming datasets or regular datasets
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                print(f"\nExample {i+1}:")
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        elif isinstance(dataset, dict):
            # For datasets with splits
            first_split_name = next(iter(dataset.keys()))
            first_split = dataset[first_split_name]
            print(f"Showing samples from '{first_split_name}' split:")
            
            for i in range(min(num_samples, len(first_split))):
                example = first_split[i]
                print(f"\nExample {i+1}:")
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
                        
    except Exception as e:
        print(f"Error displaying sample data: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace dataset: rootsautomation/RICO-Screen2Words",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_rico_dataset.py
  python download_rico_dataset.py --cache_dir ./my_datasets
  python download_rico_dataset.py --split train --streaming
  python download_rico_dataset.py --no-samples
  python download_rico_dataset.py --no-mirror  # Use official HuggingFace Hub
  python download_rico_dataset.py --mirror-url https://custom-mirror.com
        """
    )
    
    parser.add_argument(
        '--dataset-name', 
        default='rootsautomation/RICO-Screen2Words',
        help='Dataset name to download (default: rootsautomation/RICO-Screen2Words)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Directory to cache the dataset (default: HuggingFace default cache)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        help='Specific split to download (e.g., train, test, validation)'
    )
    
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Use streaming mode (no local download, good for large datasets)'
    )
    
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code execution (use with caution)'
    )
    
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip showing sample data'
    )
    
    parser.add_argument(
        '--use-mirror',
        action='store_true',
        default=True,
        help='Use HuggingFace mirror (hf-mirror.com) for faster download in China (default: True)'
    )
    
    parser.add_argument(
        '--no-mirror',
        action='store_true',
        help='Disable mirror and use official HuggingFace Hub'
    )
    
    parser.add_argument(
        '--mirror-url',
        type=str,
        default='https://hf-mirror.com',
        help='Custom mirror URL (default: https://hf-mirror.com)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=3,
        help='Number of sample examples to show (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Setup mirror configuration
    use_mirror = args.use_mirror and not args.no_mirror
    setup_hf_mirror(use_mirror=use_mirror, mirror_url=args.mirror_url)
    
    # Create cache directory if specified
    if args.cache_dir:
        cache_path = Path(args.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Using cache directory: {cache_path.absolute()}")
    
    # Check dataset info
    print("Checking dataset information...")
    if not check_dataset_info(args.dataset_name):
        print("Warning: Could not retrieve dataset information, but will try to download anyway.")
    
    # Download dataset
    dataset = download_dataset(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code
    )
    
    if dataset is None:
        print("Failed to download dataset.")
        sys.exit(1)
    
    # Show sample data
    if not args.no_samples:
        show_sample_data(dataset, args.samples)
    
    print("\n✅ Dataset download completed successfully!")
    
    if not args.streaming:
        print("\nThe dataset is now cached locally and ready for use.")
        print("You can load it in your code with:")
        print(f"from datasets import load_dataset")
        if args.cache_dir:
            print(f"dataset = load_dataset('{args.dataset_name}', cache_dir='{args.cache_dir}')")
        else:
            print(f"dataset = load_dataset('{args.dataset_name}')")

if __name__ == "__main__":
    main()