#!/usr/bin/env python3
"""
Simple merge script for Isaac Lab simulation data.
Merges pre-change and post-change transforms.json files without COLMAP processing.
"""

import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import re
from pathlib import Path

def merge_simulation_transforms(
    json_old_path: str,
    json_new_path: str,
    output_pretrain_path: str,
    output_merged_path: str,
    new_view_indices: list = None,
    split_fraction: float = 0.9,
    replace_old_views: bool = True,
    eval_on_new: bool = True
):
    """
    Merge simulation-generated transforms.json files for 3DGS-CD pipeline.
    
    Args:
        json_old_path: Path to pre-change transforms.json (rgb/ images)
        json_new_path: Path to post-change transforms.json (rgb_new/ images)
        output_pretrain_path: Output path for pre-training dataset
        output_merged_path: Output path for merged dataset
        new_view_indices: Indices of post-change views to use for training
        split_fraction: Fraction of pre-change images to use for training
        replace_old_views: Whether to replace old training views with new ones
        eval_on_new: Whether to evaluate on new views
    """
    
    # Load the transforms files
    print(f"Loading {json_old_path}")
    with open(json_old_path, 'r') as f:
        data_old = json.load(f)
    
    print(f"Loading {json_new_path}")
    with open(json_new_path, 'r') as f:
        data_new = json.load(f)




    def extract_num(fname):
        match = re.search(r'(\d+)', Path(fname).stem)
        return int(match.group(1)) if match else -1

    # Sort frames by frame number
    data_old["frames"].sort(key=lambda x: extract_num(x["file_path"]))
    data_new["frames"].sort(key=lambda x: extract_num(x["file_path"]))

        
    # Validate data compatibility
    validate_transforms_compatibility(data_old, data_new)
    
    # Create pre-training dataset (only old views)
    create_pretrain_dataset(data_old, output_pretrain_path, split_fraction)
    
    # Create merged dataset (old + new views with train/eval splits)
    create_merged_dataset(
        data_old, data_new, output_merged_path,
        new_view_indices, split_fraction, replace_old_views, eval_on_new
    )
    
    print(f"✅ Successfully created:")
    print(f"   - Pre-training dataset: {output_pretrain_path}")
    print(f"   - Merged dataset: {output_merged_path}")


def validate_transforms_compatibility(data_old, data_new):
    """Validate that old and new transforms are compatible."""
    
    # Check camera parameters match
    camera_params = ["camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h"]
    for param in camera_params:
        if param in data_old and param in data_new:
            if data_old[param] != data_new[param]:
                print(f"⚠️  Warning: Camera parameter {param} differs between datasets")
                print(f"   Old: {data_old[param]}, New: {data_new[param]}")
    
    print(f"📊 Dataset info:")
    print(f"   Pre-change frames: {len(data_old['frames'])}")
    print(f"   Post-change frames: {len(data_new['frames'])}")


def create_pretrain_dataset(data_old, output_path, split_fraction):
    """Create pre-training dataset with only old views."""
    
    # Make a copy of the old data
    pretrain_data = data_old.copy()
    
    # Create train/eval split for pre-change images
    num_images = len(data_old['frames'])
    num_train_images = math.ceil(num_images * split_fraction)
    
    # Use sequential indices starting from 0 instead of evenly spaced
    all_indices = np.arange(num_images)
    train_indices = all_indices[:num_train_images]  # First N images for training
    eval_indices = all_indices[num_train_images:]   # Remaining for eval
    
    # Get filenames
    filenames = [frame['file_path'] for frame in data_old['frames']]
    
    # Create train/val/test splits
    pretrain_data['train_filenames'] = [filenames[i] for i in train_indices]
    pretrain_data['val_filenames'] = [filenames[i] for i in eval_indices]
    pretrain_data['test_filenames'] = pretrain_data['val_filenames'].copy()
    
    # Save pre-training dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pretrain_data, f, indent=4)
    
    print(f"📋 Pre-training dataset:")
    print(f"   Train images: {len(pretrain_data['train_filenames'])}")
    print(f"   Val images: {len(pretrain_data['val_filenames'])}")
    print(f"   First train image: {pretrain_data['train_filenames'][0] if pretrain_data['train_filenames'] else 'None'}")
    print(f"   Last train image: {pretrain_data['train_filenames'][-1] if pretrain_data['train_filenames'] else 'None'}")


def create_merged_dataset(
    data_old, data_new, output_path,
    new_view_indices, split_fraction, replace_old_views, eval_on_new
):
    """Create merged dataset with both old and new views."""
    
    # Start with old data as base
    merged_data = data_old.copy()
    
    # Create initial train/eval split for old views
    num_old_images = len(data_old['frames'])
    num_train_images = math.ceil(num_old_images * split_fraction)
    
    # Use sequential indices starting from 0 instead of evenly spaced
    all_indices = np.arange(num_old_images)
    train_indices = all_indices[:num_train_images]  # First N images for training
    eval_indices = all_indices[num_train_images:]   # Remaining for eval
    
    old_filenames = [frame['file_path'] for frame in data_old['frames']]
    train_filenames = [old_filenames[i] for i in train_indices]
    eval_filenames = [old_filenames[i] for i in eval_indices]
    
    # Handle new view indices
    if new_view_indices is None:
        new_view_indices = list(range(len(data_new['frames'])))
        print(f"🔧 Using all {len(new_view_indices)} post-change views for training")
    else:
        selected_frames = [data_new['frames'][i]['file_path'] for i in new_view_indices]
        frame_numbers = [int("".join(filter(str.isdigit, Path(f).stem))) for f in selected_frames]
        print(f"🔧 Using {len(new_view_indices)} selected post-change views for training:")
        for idx, fnum, fname in zip(new_view_indices, frame_numbers, selected_frames):
            print(f"   index {idx} → frame {fnum} ({fname})")

    # Get new view filenames
    new_filenames = [data_new['frames'][i]['file_path'] for i in new_view_indices]
    all_new_filenames = [frame['file_path'] for frame in data_new['frames']]
    
    # Update training set
    if replace_old_views:
        # Replace old views with new views at MATCHING viewpoints
        print(f"🔄 Replacing old training views with new views at matching viewpoints")
        
        for new_idx in new_view_indices:
            new_filename = data_new['frames'][new_idx]['file_path']
            
            # Extract frame number from new view (e.g., "rgb_new/frame_000003.png" -> "000003")
            import re
            match = re.search(r'frame_(\d+)\.png', new_filename)
            if match:
                frame_number = match.group(1)
                # Create corresponding old view filename
                old_filename = f"rgb/frame_{frame_number}.png"
                
                # Replace if the old view is in training set
                if old_filename in train_filenames:
                    idx = train_filenames.index(old_filename)
                    train_filenames[idx] = new_filename
                    print(f"   Replaced {old_filename} with {new_filename}")
                else:
                    # Old view not in training set, add new view anyway
                    train_filenames.append(new_filename)
                    print(f"   Added {new_filename} (no matching old view in training)")
            else:
                print(f"   Warning: Could not extract frame number from {new_filename}")
                
        print(f"🔄 Replacement completed")
    else:
        # Add new views to training set
        train_filenames.extend(new_filenames)
        print(f"➕ Added {len(new_filenames)} new views to training set")
    
    # Update evaluation set
    if eval_on_new:
        # Evaluate on new views not used for training
        eval_filenames = [f for f in all_new_filenames if f not in train_filenames]
        print(f"📊 Evaluating on {len(eval_filenames)} post-change views")
    else:
        # Keep evaluating on old views
        print(f"📊 Evaluating on {len(eval_filenames)} pre-change views")
    
    # Combine all frames (old + new)
    merged_data['frames'] = data_old['frames'] + data_new['frames']
    
    # Set train/val/test splits
    merged_data['train_filenames'] = train_filenames
    merged_data['val_filenames'] = eval_filenames
    merged_data['test_filenames'] = eval_filenames.copy()
    
    # Save merged dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"📋 Merged dataset:")
    print(f"   Total frames: {len(merged_data['frames'])}")
    print(f"   Train images: {len(merged_data['train_filenames'])}")
    print(f"   Val images: {len(merged_data['val_filenames'])}")
    
    # Print training set composition
    train_old = sum(1 for f in train_filenames if f.startswith('rgb/'))
    train_new = sum(1 for f in train_filenames if f.startswith('rgb_new/'))
    print(f"   Training composition: {train_old} pre-change + {train_new} post-change")
    
    # Show first few training filenames for verification
    print(f"   First 10 training files:")
    for i, fname in enumerate(train_filenames[:10]):
        print(f"     {i}: {fname}")
    if len(train_filenames) > 10:
        print(f"     ... and {len(train_filenames) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Merge simulation transforms.json files for 3DGS-CD pipeline"
    )
    
    parser.add_argument(
        "-jo", "--json_old", type=str, required=True,
        help="Path to pre-change transforms.json"
    )
    parser.add_argument(
        "-jn", "--json_new", type=str, required=True,
        help="Path to post-change transforms.json"  
    )
    parser.add_argument(
        "-oo", "--out_json_pretrain", type=str, required=True,
        help="Output path for pre-training transforms.json"
    )
    parser.add_argument(
        "-on", "--out_json_recfg", type=str, required=True,
        help="Output path for merged transforms.json"
    )
    parser.add_argument(
        "-n", "--new_view_indices", nargs="+", type=int, default=None,
        help="Indices of post-change views to use for training (default: all)"
    )
    parser.add_argument(
        "-f", "--split_fraction", type=float, default=0.9,
        help="Fraction of pre-change images for training (default: 0.9)"
    )
    parser.add_argument(
        "--no_replace", action="store_true",
        help="Add new views to training instead of replacing old ones"
    )
    parser.add_argument(
        "--eval_old", action="store_true", 
        help="Evaluate on old views instead of new ones"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting simulation data merge...")
    print(f"📁 Input files:")
    print(f"   Pre-change: {args.json_old}")
    print(f"   Post-change: {args.json_new}")
    print(f"📁 Output files:")
    print(f"   Pre-training: {args.out_json_pretrain}")
    print(f"   Merged: {args.out_json_recfg}")
    
    merge_simulation_transforms(
        json_old_path=args.json_old,
        json_new_path=args.json_new,
        output_pretrain_path=args.out_json_pretrain,
        output_merged_path=args.out_json_recfg,
        new_view_indices=args.new_view_indices,
        split_fraction=args.split_fraction,
        replace_old_views=not args.no_replace,
        eval_on_new=not args.eval_old
    )
    
    print("✅ Merge completed successfully!")


if __name__ == "__main__":
    main()