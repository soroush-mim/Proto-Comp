#!/usr/bin/env python3
"""
Analyze PCN dataset and Cap3D captions to find missing instances and captions.

This script compares the PCN dataset instances with the Cap3D_automated_ShapeNet.csv file
to identify:
1. PCN instances without captions in the CSV
2. CSV captions without corresponding PCN instances
3. Count instances with/without captions for each split (test, train, val)
"""

import os
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


def load_caption_data(csv_path):
    """
    Load caption data from CSV file.
    
    Args:
        csv_path (str): Path to the Cap3D CSV file
        
    Returns:
        dict: Dictionary mapping 'class_id_instance_id' to caption
        set: Set of all class_ids in the CSV
    """
    print(f"Loading caption data from {csv_path}...")
    
    captions = {}
    class_ids = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) >= 2:
                    instance_id = row[0].strip()
                    caption = row[1].strip()
                    captions[instance_id] = caption
                    
                    # Extract class_id from instance_id
                    class_id = instance_id.split('_')[0]
                    class_ids.add(class_id)
        
        print(f"✓ Loaded {len(captions)} captions for {len(class_ids)} classes")
        return captions, class_ids
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return {}, set()


def get_pcn_instances(pcn_base_path):
    """
    Get all PCN instances from the complete folders.
    
    Args:
        pcn_base_path (str): Path to the PCN dataset base directory
        
    Returns:
        dict: Dictionary with structure {split: {class_id: [instance_ids]}}
        set: Set of all class_ids in PCN
    """
    print(f"Scanning PCN dataset at {pcn_base_path}...")
    
    pcn_instances = {'test': {}, 'train': {}, 'val': {}}
    all_class_ids = set()
    
    splits = ['test', 'train', 'val']
    
    for split in splits:
        complete_path = Path(pcn_base_path) / split / 'complete'
        
        if not complete_path.exists():
            print(f"⚠️  Warning: {complete_path} does not exist")
            continue
            
        print(f"  Processing {split} split...")
        
        # Get all class directories
        class_dirs = [d for d in complete_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_id = class_dir.name
            all_class_ids.add(class_id)
            
            # Get all .pcd files in this class directory
            pcd_files = list(class_dir.glob('*.pcd'))
            instance_ids = [f.stem for f in pcd_files]  # filename without extension
            
            pcn_instances[split][class_id] = instance_ids
            
            print(f"    Class {class_id}: {len(instance_ids)} instances")
    
    print(f"✓ Found {len(all_class_ids)} classes in PCN dataset")
    return pcn_instances, all_class_ids


def analyze_missing_captions(pcn_instances, captions):
    """
    Find PCN instances without captions and vice versa.
    
    Args:
        pcn_instances (dict): PCN instances by split and class
        captions (dict): Caption data
        
    Returns:
        dict: Analysis results
    """
    print("\nAnalyzing missing captions...")
    
    results = {
        'pcn_without_caption': {'test': {}, 'train': {}, 'val': {}},
        'caption_without_pcn': {},
        'stats_by_split': {'test': {}, 'train': {}, 'val': {}},
        'total_stats': {}
    }
    
    # Find PCN instances without captions
    for split in ['test', 'train', 'val']:
        for class_id, instance_ids in pcn_instances[split].items():
            missing_captions = []
            with_captions = []
            
            for instance_id in instance_ids:
                full_instance_id = f"{class_id}_{instance_id}"
                
                if full_instance_id in captions:
                    with_captions.append(instance_id)
                else:
                    missing_captions.append(instance_id)
            
            results['pcn_without_caption'][split][class_id] = missing_captions
            results['stats_by_split'][split][class_id] = {
                'total': len(instance_ids),
                'with_caption': len(with_captions),
                'without_caption': len(missing_captions)
            }
    
    # Find captions without PCN instances
    all_pcn_instances = set()
    for split in ['test', 'train', 'val']:
        for class_id, instance_ids in pcn_instances[split].items():
            for instance_id in instance_ids:
                all_pcn_instances.add(f"{class_id}_{instance_id}")
    
    for instance_id in captions.keys():
        if instance_id not in all_pcn_instances:
            class_id = instance_id.split('_')[0]
            if class_id not in results['caption_without_pcn']:
                results['caption_without_pcn'][class_id] = []
            results['caption_without_pcn'][class_id].append(instance_id)
    
    return results


def print_detailed_report(results, captions, class_ids_csv, class_ids_pcn):
    """
    Print detailed analysis report.
    """
    print("\n" + "="*80)
    print("PCN DATASET vs CAP3D CAPTIONS ANALYSIS REPORT")
    print("="*80)
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"  • Total captions in CSV: {len(captions):,}")
    print(f"  • Classes in CSV: {len(class_ids_csv)}")
    print(f"  • Classes in PCN: {len(class_ids_pcn)}")
    print(f"  • Common classes: {len(class_ids_csv & class_ids_pcn)}")
    print(f"  • Classes only in CSV: {len(class_ids_csv - class_ids_pcn)}")
    print(f"  • Classes only in PCN: {len(class_ids_pcn - class_ids_csv)}")
    
    # Captions without PCN instances
    total_captions_without_pcn = sum(len(instances) for instances in results['caption_without_pcn'].values())
    print(f"  • Captions without PCN instances: {total_captions_without_pcn:,}")
    
    # Statistics by split
    print("\n📈 STATISTICS BY SPLIT:")
    total_with_caption = 0
    total_without_caption = 0
    total_instances = 0
    
    for split in ['test', 'train', 'val']:
        split_with_caption = 0
        split_without_caption = 0
        split_total = 0
        
        for class_id, stats in results['stats_by_split'][split].items():
            split_with_caption += stats['with_caption']
            split_without_caption += stats['without_caption']
            split_total += stats['total']
        
        total_with_caption += split_with_caption
        total_without_caption += split_without_caption
        total_instances += split_total
        
        print(f"\n  {split.upper()} SPLIT:")
        print(f"    • Total instances: {split_total:,}")
        print(f"    • With captions: {split_with_caption:,} ({split_with_caption/split_total*100:.1f}%)")
        print(f"    • Without captions: {split_without_caption:,} ({split_without_caption/split_total*100:.1f}%)")
    
    print(f"\n  OVERALL:")
    print(f"    • Total instances: {total_instances:,}")
    print(f"    • With captions: {total_with_caption:,} ({total_with_caption/total_instances*100:.1f}%)")
    print(f"    • Without captions: {total_without_caption:,} ({total_without_caption/total_instances*100:.1f}%)")
    
    # Detailed breakdown by class
    print("\n📋 DETAILED BREAKDOWN BY CLASS:")
    print(f"{'Class ID':<12} {'Test':<8} {'Train':<8} {'Val':<8} {'Total':<8} {'With Caption':<12} {'Missing':<12}")
    print("-" * 80)
    
    all_classes = set()
    for split in ['test', 'train', 'val']:
        all_classes.update(results['stats_by_split'][split].keys())
    
    for class_id in sorted(all_classes):
        test_stats = results['stats_by_split']['test'].get(class_id, {'total': 0, 'with_caption': 0, 'without_caption': 0})
        train_stats = results['stats_by_split']['train'].get(class_id, {'total': 0, 'with_caption': 0, 'without_caption': 0})
        val_stats = results['stats_by_split']['val'].get(class_id, {'total': 0, 'with_caption': 0, 'without_caption': 0})
        
        total_instances = test_stats['total'] + train_stats['total'] + val_stats['total']
        total_with_caption = test_stats['with_caption'] + train_stats['with_caption'] + val_stats['with_caption']
        total_without_caption = test_stats['without_caption'] + train_stats['without_caption'] + val_stats['without_caption']
        
        if total_instances > 0:  # Only show classes that exist in PCN
            print(f"{class_id:<12} {test_stats['total']:<8} {train_stats['total']:<8} {val_stats['total']:<8} {total_instances:<8} {total_with_caption:<12} {total_without_caption:<12}")
    
    # Classes only in CSV
    if results['caption_without_pcn']:
        print(f"\n⚠️  CLASSES WITH CAPTIONS BUT NO PCN INSTANCES:")
        for class_id, instances in results['caption_without_pcn'].items():
            print(f"    {class_id}: {len(instances)} captions")
    
    # Classes only in PCN
    pcn_only_classes = class_ids_pcn - class_ids_csv
    if pcn_only_classes:
        print(f"\n⚠️  CLASSES IN PCN BUT NOT IN CSV:")
        for class_id in sorted(pcn_only_classes):
            total_instances = 0
            for split in ['test', 'train', 'val']:
                if class_id in results['stats_by_split'][split]:
                    total_instances += results['stats_by_split'][split][class_id]['total']
            print(f"    {class_id}: {total_instances} instances")


def save_detailed_results(results, output_dir):
    """
    Save detailed results to files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save instances without captions by split
    for split in ['test', 'train', 'val']:
        filename = output_path / f"pcn_without_captions_{split}.txt"
        with open(filename, 'w') as f:
            f.write(f"PCN instances without captions - {split} split\n")
            f.write("="*50 + "\n\n")
            
            for class_id, instances in results['pcn_without_caption'][split].items():
                if instances:
                    f.write(f"Class {class_id} ({len(instances)} instances):\n")
                    for instance in instances:
                        f.write(f"  {class_id}_{instance}\n")
                    f.write("\n")
    
    # Save captions without PCN instances
    if results['caption_without_pcn']:
        filename = output_path / "captions_without_pcn.txt"
        with open(filename, 'w') as f:
            f.write("Captions without corresponding PCN instances\n")
            f.write("="*50 + "\n\n")
            
            for class_id, instances in results['caption_without_pcn'].items():
                f.write(f"Class {class_id} ({len(instances)} captions):\n")
                for instance in instances:
                    f.write(f"  {instance}\n")
                f.write("\n")
    
    print(f"\n💾 Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PCN dataset and Cap3D captions')
    parser.add_argument('--csv_path', type=str, 
                       default='data/PCN/Cap3D_automated_ShapeNet.csv',
                       help='Path to Cap3D CSV file')
    parser.add_argument('--pcn_path', type=str,
                       default='data/PCN',
                       help='Path to PCN dataset directory')
    parser.add_argument('--output_dir', type=str,
                       default='analysis_results',
                       help='Directory to save detailed results')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to files')
    
    args = parser.parse_args()
    
    # Load data
    captions, class_ids_csv = load_caption_data(args.csv_path)
    pcn_instances, class_ids_pcn = get_pcn_instances(args.pcn_path)
    
    if not captions or not pcn_instances:
        print("❌ Failed to load data. Please check file paths.")
        return
    
    # Analyze
    results = analyze_missing_captions(pcn_instances, captions)
    
    # Print report
    print_detailed_report(results, captions, class_ids_csv, class_ids_pcn)
    
    # Save results if requested
    if args.save_results:
        save_detailed_results(results, args.output_dir)


if __name__ == "__main__":
    main()
