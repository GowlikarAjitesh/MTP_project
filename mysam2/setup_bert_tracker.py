#!/usr/bin/env python3
"""
Checkpoint finder and setup utility for BERT Tracker
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_checkpoint(model_name="sam2_hiera_l"):
    """Find SAM2 checkpoint in common locations"""
    
    possible_paths = [
        "/home/cs24m118/sam2_backup/checkpoints",
        "/home/cs24m118/phase2/sam2/checkpoints",
        "/home/cs24m118/phase2/DAM4SAM/checkpoints",
        os.path.expanduser("~/.cache/sam2"),
        "/root/.cache/sam2",
        "/workspace/sam2/checkpoints",
        "/root/sam2/checkpoints",
    ]
    
    logger.info(f"Searching for {model_name} checkpoint...")
    
    for base_path in possible_paths:
        base_path = Path(base_path)
        if not base_path.exists():
            continue
        
        # Try exact match
        checkpoint_path = base_path / f"{model_name}.pt"
        if checkpoint_path.exists():
            logger.info(f"✓ Found: {checkpoint_path}")
            return str(checkpoint_path)
        
        # Try with different extensions
        for pattern in [f"{model_name}*", "*.pt", "sam2*.pt"]:
            matches = list(base_path.glob(pattern))
            if matches:
                logger.info(f"Found candidates in {base_path}:")
                for m in matches:
                    logger.info(f"  - {m.name}")
                if len(matches) == 1:
                    logger.info(f"✓ Selected: {matches[0]}")
                    return str(matches[0])
    
    logger.warning(f"Checkpoint not found for {model_name}")
    logger.info("\nTried locations:")
    for p in possible_paths:
        logger.info(f"  - {p}")
    
    return None


def check_dataset():
    """Check MOT17 dataset structure"""
    dataset_root = Path("/home/cs24m118/datasets/MOT17")
    
    logger.info("\nChecking MOT17 dataset structure...")
    
    if not dataset_root.exists():
        logger.error(f"Dataset not found at {dataset_root}")
        return False
    
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    
    if train_dir.exists():
        sequences = list(train_dir.iterdir())
        logger.info(f"✓ Train split: {len(sequences)} sequences")
        for seq in sorted(sequences)[:3]:
            img_count = len(list((seq / "img1").glob("*.jpg"))) if (seq / "img1").exists() else 0
            logger.info(f"    {seq.name}: {img_count} frames")
    
    if test_dir.exists():
        sequences = list(test_dir.iterdir())
        logger.info(f"✓ Test split: {len(sequences)} sequences")
    
    return True


def setup_environment():
    """Setup Python environment"""
    logger.info("\nSetting up environment...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not installed")
        return False
    
    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV not installed")
        return False
    
    try:
        import numpy
        logger.info(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        logger.error("NumPy not installed")
        return False
    
    return True


def main():
    logger.info("="*60)
    logger.info("BERT Tracker Setup Utility")
    logger.info("="*60)
    
    # Check environment
    if not setup_environment():
        logger.error("Environment setup failed")
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        logger.error("Dataset check failed")
        sys.exit(1)
    
    # Find checkpoint
    checkpoint = find_checkpoint("sam2_hiera_l")
    
    if checkpoint is None:
        logger.warning("\nNo checkpoint found. You can:")
        logger.info("1. Download from: https://github.com/facebookresearch/segment-anything-2")
        logger.info("2. Place in: /home/cs24m118/sam2_backup/checkpoints/")
        logger.info("3. Or specify with --checkpoint flag")
    
    logger.info("\n" + "="*60)
    logger.info("Setup Complete!")
    logger.info("="*60)
    logger.info("\nTo run benchmark:")
    logger.info("  python run_mot17_benchmark.py --split train --sequences MOT17-02-FRCNN")
    logger.info("\nTo process all sequences:")
    logger.info("  python run_mot17_benchmark.py --split train")
    logger.info("\nWith custom checkpoint:")
    logger.info("  python run_mot17_benchmark.py --split train --checkpoint /path/to/checkpoint.pt")


if __name__ == "__main__":
    main()
