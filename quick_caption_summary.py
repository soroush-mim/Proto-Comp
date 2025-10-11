#!/usr/bin/env python3
"""
Quick summary script for PCN caption analysis.
Provides a concise overview of caption coverage.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict


def quick_summary(csv_path="data/PCN/Cap3D_automated_ShapeNet.csv", 
                  pcn_path="data/PCN"):
    """
    Generate a quick summary of caption coverage.
    """
    print("PCN Dataset Caption Coverage Summary")
    print("=" * 50)
    
    # Load captions
    captions = {}
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) >= 2:
                captions[row[0].strip()] = row[1].strip()
    
    # Count PCN instances
    pcn_counts = {'test': 0, 'train': 0, 'val': 0}
    pcn_with_captions = {'test': 0, 'train': 0, 'val': 0}
    
    for split in ['test', 'train', 'val']:
        complete_path = Path(pcn_path) / split / 'complete'
        if complete_path.exists():
            for class_dir in complete_path.iterdir():
                if class_dir.is_dir():
                    class_id = class_dir.name
                    pcd_files = list(class_dir.glob('*.pcd'))
                    
                    for pcd_file in pcd_files:
                        instance_id = pcd_file.stem
                        full_id = f"{class_id}_{instance_id}"
                        
                        pcn_counts[split] += 1
                        if full_id in captions:
                            pcn_with_captions[split] += 1
    
    # Print summary
    total_pcn = sum(pcn_counts.values())
    total_with_captions = sum(pcn_with_captions.values())
    
    print(f"Total PCN instances: {total_pcn:,}")
    print(f"With captions: {total_with_captions:,} ({total_with_captions/total_pcn*100:.1f}%)")
    print(f"Without captions: {total_pcn-total_with_captions:,} ({(total_pcn-total_with_captions)/total_pcn*100:.1f}%)")
    print()
    
    for split in ['test', 'train', 'val']:
        coverage = pcn_with_captions[split]/pcn_counts[split]*100 if pcn_counts[split] > 0 else 0
        print(f"{split.capitalize()}: {pcn_with_captions[split]:,}/{pcn_counts[split]:,} ({coverage:.1f}%)")
    
    print(f"\nTotal captions in CSV: {len(captions):,}")


if __name__ == "__main__":
    quick_summary()
