#!/usr/bin/env python3
"""
Test localization pipeline.
"""

from pathlib import Path
# from localization import reconstruct_scene, relocalize_images
from localization.reconstruction import reconstruct_scene
from localization.relocalization import relocalize_images


def main():
    """Run full reconstruction and relocalization pipeline."""
    
    # Define paths
    pre_scene_dir = Path('data_real/day8/pre')
    post_scene_dir = Path('data_real/day8/post')
    
    print("=" * 80)
    print("LOCALIZATION PIPELINE")
    print("=" * 80)
    print()
    
    print("Reconstructing reference scene...")
    print()
    
    recon_result = reconstruct_scene(
        scene_dir=str(pre_scene_dir),
        num_ba_iterations=2,
        max_keypoints=8192,
        match_threshold=0.2,
        num_matched=10
    )
    
    print()
    

    print("Relocalizing query images...")
    print()
    
    reloc_result = relocalize_images(
        pre_dir=str(pre_scene_dir),
        post_dir=str(post_scene_dir),
        num_matched=10,
        ransac_max_error=2.0
    )
    
    print()
    
    print("\n🎉 LOCALIZATION COMPLETE")
    print()
    
    print("RECONSTRUCTION:")
    print(f"  Reference model: {recon_result.colmap_ba_dir}")
    print(f"  Images: {recon_result.num_images}")
    print(f"  3D Points: {recon_result.num_points3d}")
    print(f"  Refined transforms: {recon_result.transforms_path}")
    print()
    
    print("RELOCALIZATION:")
    print(f"  Query images: {reloc_result.num_query_images}")
    print(f"  Localized: {reloc_result.num_localized} ({reloc_result.success_rate:.1%})")
    print(f"  Output transforms: {reloc_result.output_transforms_path}")
    print()
    
    print("Query images now in reference coordinate frame!")
    print()


if __name__ == '__main__':
    main()