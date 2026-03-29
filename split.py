#!/usr/bin/env python3
"""
Split barley and wheat datasets into train/val with multiple percentage ratios.

Folder structure created:
    Stomata_Dataset/
      ├── BARLEY/
      │   ├── 1%/
      │   │   ├── images/
      │   │   │   ├── train/
      │   │   │   └── val/
      │   │   └── labels/
      │   │       ├── train/
      │   │       └── val/
      │   ├── 5%/
      │   ├── 10%/
      │   └── 20%/
      └── WHEAT/
          ├── 1%/
          ├── 5%/
          ├── 10%/
          └── 20%/
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import random
from collections import defaultdict


class DatasetSplitter:
    """Split datasets by crop type and percentage with train/val split."""
    
    def __init__(
        self,
        all_image_dir: str,
        all_label_dir: str,
        output_root: str,
        seed: int = 42,
    ):
        """
        Args:
            all_image_dir: Path to ALL_DATA_IMAGE/images
            all_label_dir: Path to ALL_DATA_LABEL/labels
            output_root: Path to Stomata_Dataset
            seed: Random seed for reproducibility
        """
        self.all_image_dir = Path(all_image_dir)
        self.all_label_dir = Path(all_label_dir)
        self.output_root = Path(output_root)
        self.seed = seed
        random.seed(seed)
        
        self.image_ext = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
        self.label_ext = {'.txt', '.xml'}
    
    def get_image_files(self, crop_type: str) -> List[str]:
        """Get all image filenames for a crop type (without extension)."""
        image_dir = self.all_image_dir / f"{crop_type}"
        if not image_dir.exists():
            print(f"❌ Not found: {image_dir}")
            return []
        
        files = set()
        for f in image_dir.iterdir():
            if f.is_file() and f.suffix in self.image_ext:
                files.add(f.stem)  # Get filename without extension
        
        return sorted(list(files))
    
    def split_by_percentage(
        self, file_list: List[str], train_percentage: float
    ) -> Tuple[List[str], List[str]]:
        """
        Split file list: train_percentage% for training, rest for validation.
        
        Args:
            file_list: All files
            train_percentage: Percentage to use for training (1, 5, 10, 20)
        
        Returns:
            (train_files, val_files)
        """
        shuffled = file_list.copy()
        random.shuffle(shuffled)
        
        split_point = int(len(shuffled) * train_percentage / 100.0)
        train_files = shuffled[:split_point]
        val_files = shuffled[split_point:]
        
        return sorted(train_files), sorted(val_files)
    
    def copy_file_with_fallback(self, src_dir: Path, dst_dir: Path, filename: str, extensions: set):
        """
        Copy a file trying multiple extensions.
        
        Args:
            src_dir: Source directory
            dst_dir: Destination directory
            filename: Base filename (without extension)
            extensions: Set of possible extensions to try
        """
        for ext in extensions:
            src_file = src_dir / f"{filename}{ext}"
            if src_file.exists():
                dst_file = dst_dir / f"{filename}{ext}"
                shutil.copy2(src_file, dst_file)
                return True
        return False
    
    def process_crop(self, crop_type: str, percentages: List[float]):
        """
        Process a single crop type (barley or wheat).
        
        Args:
            crop_type: 'barley195' or 'wheat522'
            percentages: List of percentages to create [1, 5, 10, 20]
        """
        print(f"\n📁 Processing {crop_type}...")
        
        # Get all images for this crop
        all_files = self.get_image_files(crop_type)
        if not all_files:
            print(f"❌ No images found for {crop_type}")
            return
        
        print(f"   Total files: {len(all_files)}")
        
        # Determine output crop folder name (BARLEY or WHEAT)
        if "barley" in crop_type.lower():
            crop_folder = "BARLEY"
            src_image_dir = self.all_image_dir / crop_type
            src_label_dir = self.all_label_dir / crop_type
        else:
            crop_folder = "WHEAT"
            src_image_dir = self.all_image_dir / crop_type
            src_label_dir = self.all_label_dir / crop_type
        
        # Process each percentage
        for pct in percentages:
            print(f"   Creating {pct}% training subset...")
            
            # Split: pct% for training, rest for validation
            train_files, val_files = self.split_by_percentage(all_files, pct)
            print(f"     Train: {len(train_files)} ({pct}%), Val: {len(val_files)} ({100-pct}%)")
            
            # Create directory structure
            pct_str = f"{int(pct)}%"
            base_dir = self.output_root / crop_folder / pct_str
            
            # Image directories
            train_img_dir = base_dir / "images" / "train"
            val_img_dir = base_dir / "images" / "val"
            train_img_dir.mkdir(parents=True, exist_ok=True)
            val_img_dir.mkdir(parents=True, exist_ok=True)
            
            # Label directories
            train_lbl_dir = base_dir / "labels" / "train"
            val_lbl_dir = base_dir / "labels" / "val"
            train_lbl_dir.mkdir(parents=True, exist_ok=True)
            val_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy training files
            for fname in train_files:
                self.copy_file_with_fallback(src_image_dir, train_img_dir, fname, self.image_ext)
                self.copy_file_with_fallback(src_label_dir, train_lbl_dir, fname, self.label_ext)
            
            # Copy validation files
            for fname in val_files:
                self.copy_file_with_fallback(src_image_dir, val_img_dir, fname, self.image_ext)
                self.copy_file_with_fallback(src_label_dir, val_lbl_dir, fname, self.label_ext)
            
            print(f"     ✅ Created: {base_dir}")
    
    def run(self, percentages: List[float] = [1, 5, 10, 20]):
        """
        Run the full split pipeline.
        
        Args:
            percentages: List of percentages to create [1, 5, 10, 20]
        """
        print("=" * 70)
        print("🌾 Starting Dataset Split: Barley & Wheat with Multiple Percentages")
        print("=" * 70)
        
        # Create output root if it doesn't exist
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Process barley and wheat
        self.process_crop("barley195", percentages)
        self.process_crop("wheat522", percentages)
        
        print("\n" + "=" * 70)
        print("✅ Dataset split complete!")
        print("=" * 70)
        print(f"\nOutput structure (X% = training, rest = validation):")
        print(f"  {self.output_root}/BARLEY/")
        for pct in percentages:
            print(f"    {pct}%/")
            print(f"      ├── images/ (train/: {pct}%, val/: {100-pct}%)")
            print(f"      └── labels/ (train/: {pct}%, val/: {100-pct}%)")
        print(f"  {self.output_root}/WHEAT/")
        for pct in percentages:
            print(f"    {pct}%/")
            print(f"      ├── images/ (train/: {pct}%, val/: {100-pct}%)")
            print(f"      └── labels/ (train/: {pct}%, val/: {100-pct}%)")


def print_dataset_info(root_dir: str):
    """Print summary of created datasets."""
    root = Path(root_dir)
    
    if not root.exists():
        return
    
    print("\n" + "=" * 70)
    print("📊 Dataset Summary")
    print("=" * 70)
    
    for crop in ["BARLEY", "WHEAT"]:
        crop_dir = root / crop
        if not crop_dir.exists():
            continue
        
        print(f"\n{crop}:")
        for pct_dir in sorted(crop_dir.iterdir()):
            if pct_dir.is_dir():
                train_img = len(list((pct_dir / "images" / "train").glob("*")))
                val_img = len(list((pct_dir / "images" / "val").glob("*")))
                print(f"  {pct_dir.name:5s}: {train_img:3d} train, {val_img:3d} val")


if __name__ == "__main__":
    import sys
    
    # Paths
    project_root = Path(__file__).parent
    all_image_dir = project_root / "Stomata_Dataset" / "ALL_DATA_IMAGE" / "images"
    all_label_dir = project_root / "Stomata_Dataset" / "ALL_DATA_LABEL" / "labels"
    output_root = project_root / "Stomata_Dataset"
    
    # Configuration
    percentages = [1, 5, 10, 20]  # Percentages for training (rest = validation)
    seed = 42
    
    # Run splitter
    splitter = DatasetSplitter(
        all_image_dir=str(all_image_dir),
        all_label_dir=str(all_label_dir),
        output_root=str(output_root),
        seed=seed,
    )
    
    splitter.run(percentages=percentages)
    print_dataset_info(str(output_root))
