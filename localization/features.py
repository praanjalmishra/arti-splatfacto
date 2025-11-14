"""
Feature extraction and matching utilities. Wraps HLoc functionality.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
from collections import defaultdict

from hloc import extract_features, match_features, pairs_from_retrieval, pairs_from_poses
from hloc.utils.io import list_h5_names
import pycolmap


def extract_local_features(
    image_dir: Path,
    output_h5: Path,
    feature_conf: Dict,
    image_list: Optional[List[str]] = None
) -> Path:
    """
    Extract local features (SuperPoint) from images.
    
    Args:
        image_dir: Directory containing images
        output_h5: Output path for features.h5
        feature_conf: Feature extraction config dict
        image_list: Optional list of image names to process (relative to image_dir)
    
    Returns:
        Path to the generated features.h5 file
    """
    if image_list is None:
        # Get all images in directory
        image_list = [
            p.relative_to(image_dir).as_posix() 
            for p in image_dir.iterdir() 
            if p.suffix.lower() in ['.jpg', '.png', '.jpeg']
        ]
    
    print(f"Extracting local features from {len(image_list)} images...")
    
    extract_features.main(
        feature_conf,
        image_dir,
        image_list=image_list,
        feature_path=output_h5
    )
    
    print(f"Features saved to: {output_h5}")
    return output_h5


def extract_global_descriptors(
    image_dir: Path,
    output_h5: Path,
    global_conf: Dict,
    export_dir: Optional[Path] = None
) -> Path:
    """
    Extract global descriptors (NetVLAD) from images.
    
    Args:
        image_dir: Directory containing images
        output_h5: Output path for global-descriptors.h5
        global_conf: Global descriptor config dict
        export_dir: Export directory (defaults to output_h5 parent)
    
    Returns:
        Path to the generated global-descriptors.h5 file
    """
    if export_dir is None:
        export_dir = output_h5.parent
    
    print(f"Extracting global descriptors...")
    
    extract_features.main(
        conf=global_conf,
        image_dir=image_dir,
        export_dir=export_dir,
        feature_path=output_h5
    )
    
    print(f"✓ Global descriptors saved to: {output_h5}")
    return output_h5


def generate_pairs_from_poses(
    colmap_model: Path,
    output_pairs: Path,
    num_matched: int = 10
) -> Path:
    """
    Generate image pairs based on camera poses proximity.
    Good for sequential/video data.
    
    Args:
        colmap_model: Path to COLMAP model directory
        output_pairs: Output path for pairs file
        num_matched: Number of nearest neighbors to match
    
    Returns:
        Path to the generated pairs file
    """
    print(f"Generating pairs from poses (k={num_matched})...")
    
    pairs_from_poses.main(colmap_model, output_pairs, num_matched)
    
    print(f"✓ Pairs saved to: {output_pairs}")
    return output_pairs


def generate_pairs_from_retrieval(
    query_descriptors: Path,
    reference_descriptors: Path,
    output_pairs: Path,
    reference_model: Optional[Path] = None,
    num_matched: int = 10
) -> Dict[str, List[str]]:
    """
    Generate image pairs using global descriptor similarity.
    Returns both pairs file and dictionary.
    
    Args:
        query_descriptors: Query global descriptors h5
        reference_descriptors: Reference global descriptors h5
        output_pairs: Output path for pairs file
        reference_model: Optional COLMAP model for reference images
        num_matched: Number of top matches per query
    
    Returns:
        Dictionary mapping query image names to list of reference image names
    """
    print(f"Generating pairs from retrieval (k={num_matched})...")
    
    # Load image names
    query_names = list_h5_names(query_descriptors)
    
    if reference_model is not None:
        # Load reference names from COLMAP model
        reconstruction = pycolmap.Reconstruction(str(reference_model))
        reference_names = [img.name for img in reconstruction.images.values()]
    else:
        # Load reference names from descriptors
        reference_names = list_h5_names(reference_descriptors)
    
    # Get descriptors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    name2db = {n: 0 for n in list_h5_names(reference_descriptors)}
    ref_desc = pairs_from_retrieval.get_descriptors(reference_names, [reference_descriptors], name2db)
    query_desc = pairs_from_retrieval.get_descriptors(query_names, query_descriptors)
    
    # Compute similarity matrix
    sim = torch.einsum('id,jd->ij', query_desc.to(device), ref_desc.to(device))
    
    # Get top matches
    pairs = pairs_from_retrieval.pairs_from_score_matrix(
        sim, 
        np.zeros((len(query_names), len(reference_names)), bool),
        num_matched, 
        min_score=0
    )
    
    # Convert to dictionary
    pairs_dict = defaultdict(list)
    for i, j in pairs:
        pairs_dict[query_names[i]].append(reference_names[j])
    
    # Write pairs file
    with open(output_pairs, 'w') as f:
        for query_name, ref_names in pairs_dict.items():
            for ref_name in ref_names:
                f.write(f'{query_name} {ref_name}\n')
    
    print(f"✓ Generated {len(pairs)} pairs for {len(query_names)} queries")
    print(f"✓ Pairs saved to: {output_pairs}")
    
    return dict(pairs_dict)


def match_feature_pairs(
    pairs_file: Path,
    features_query: Path,
    output_matches: Path,
    matcher_conf: Dict,
    features_ref: Optional[Path] = None,
    export_dir: Optional[Path] = None
) -> Path:
    """
    Match features between image pairs.
    
    Args:
        pairs_file: Text file with image pairs (one pair per line)
        features_query: Query features h5 file
        output_matches: Output path for matches.h5
        matcher_conf: Matcher config dict
        features_ref: Reference features h5 (if different from query)
        export_dir: Export directory (defaults to output_matches parent)
    
    Returns:
        Path to the generated matches.h5 file
    """
    if export_dir is None:
        export_dir = output_matches.parent
    
    # Count pairs
    with open(pairs_file, 'r') as f:
        num_pairs = sum(1 for _ in f)
    
    print(f"Matching {num_pairs} image pairs...")
    
    if features_ref is not None:
        # Cross-matching (query vs reference)
        match_features.main(
            matcher_conf,
            pairs_file,
            features=features_query,
            features_ref=features_ref,
            export_dir=export_dir,
            matches=output_matches
        )
    else:
        # Self-matching (within same feature set)
        match_features.main(
            matcher_conf,
            pairs_file,
            features=features_query,
            export_dir=export_dir,
            matches=output_matches
        )
    
    print(f"✓ Matches saved to: {output_matches}")
    return output_matches


def get_image_list(image_dir: Path, extensions: List[str] = None) -> List[str]:
    """
    Get list of image filenames in directory.
    
    Args:
        image_dir: Directory to search
        extensions: List of valid extensions (default: ['.jpg', '.png', '.jpeg'])
    
    Returns:
        List of image names relative to image_dir
    """
    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg']
    
    extensions = [ext.lower() for ext in extensions]
    
    images = [
        p.relative_to(image_dir).as_posix() 
        for p in sorted(image_dir.iterdir()) 
        if p.suffix.lower() in extensions
    ]
    
    return images


# High-level convenience functions
def extract_all_features(
    image_dir: Path,
    output_dir: Path,
    feature_conf: Dict,
    global_conf: Dict,
    image_list: Optional[List[str]] = None
) -> Tuple[Path, Path]:
    """
    Extract both local and global features in one call.
    
    Returns:
        Tuple of (features_h5_path, global_descriptors_h5_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    features_h5 = output_dir / 'features.h5'
    global_h5 = output_dir / 'global-descriptors.h5'
    
    # Extract local features
    extract_local_features(image_dir, features_h5, feature_conf, image_list)
    
    # Extract global descriptors
    extract_global_descriptors(image_dir, global_h5, global_conf, output_dir)
    
    return features_h5, global_h5


def match_for_reconstruction(
    image_dir: Path,
    colmap_model: Path,
    features_h5: Path,
    output_dir: Path,
    matcher_conf: Dict,
    num_matched: int = 10
) -> Tuple[Path, Path]:
    """
    Generate pairs from poses and match features for reconstruction.
    
    Returns:
        Tuple of (pairs_file_path, matches_h5_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    pairs_file = output_dir / 'pairs-sfm.txt'
    matches_h5 = output_dir / 'matches.h5'
    
    # Generate pairs from poses
    generate_pairs_from_poses(colmap_model, pairs_file, num_matched)
    
    # Match features
    match_feature_pairs(pairs_file, features_h5, matches_h5, matcher_conf)
    
    return pairs_file, matches_h5


def match_for_localization(
    query_images_dir: Path,
    query_features_h5: Path,
    query_global_h5: Path,
    reference_features_h5: Path,
    reference_global_h5: Path,
    reference_model: Path,
    output_dir: Path,
    matcher_conf: Dict,
    num_matched: int = 10
) -> Tuple[Path, Path, Dict[str, List[str]]]:
    """
    Generate pairs from retrieval and match features for localization.
    
    Returns:
        Tuple of (pairs_file_path, matches_h5_path, pairs_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    pairs_file = output_dir / 'pairs-localization.txt'
    matches_h5 = output_dir / 'matches.h5'
    
    # Generate pairs from retrieval
    pairs_dict = generate_pairs_from_retrieval(
        query_global_h5,
        reference_global_h5,
        pairs_file,
        reference_model,
        num_matched
    )
    
    # Match features (cross-matching)
    match_feature_pairs(
        pairs_file,
        query_features_h5,
        matches_h5,
        matcher_conf,
        features_ref=reference_features_h5,
        export_dir=output_dir
    )
    
    return pairs_file, matches_h5, pairs_dict