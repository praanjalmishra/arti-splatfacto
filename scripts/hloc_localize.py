"""
Localize post-change images against pre-change scene reconstruction
"""

from pathlib import Path
import json
import numpy as np
import pycolmap
import h5py
from scipy.spatial.transform import Rotation
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer
from hloc.utils.io import get_keypoints
from collections import defaultdict

def localize_post_images(
    pre_sfm_dir,           # Path to pre-change reconstruction (from script 1)
    post_image_dir,        # Path to post-change images
    post_transforms_file,  # Where to save results
    num_retrieval=10,      # Number of reference images to match
    ransac_thresh=12.0     # RANSAC threshold in pixels
):
    """
    Localize post-change images against pre-change reconstruction
    
    Args:
        pre_sfm_dir: Directory with pre-change hloc outputs (contains sfm/, features.h5, etc)
        post_image_dir: Directory with post-change frames
        post_transforms_file: Output path for transforms.json
        num_retrieval: Number of pre-change images to retrieve per query
        ransac_thresh: RANSAC threshold
    """
    
    pre_sfm_dir = Path(pre_sfm_dir)
    post_image_dir = Path(post_image_dir)
    post_transforms_file = Path(post_transforms_file)
    
    # Output paths for post-change features
    post_outputs = post_image_dir.parent / "post_hloc_outputs"
    post_outputs.mkdir(exist_ok=True)
    
    post_features = post_outputs / "post_features.h5"
    post_global = post_outputs / "post_global.h5"
    post_matches = post_outputs / "post_matches.h5"
    pairs_file = post_outputs / "post_pairs.txt"
    
    # Paths to pre-change data
    pre_features = pre_sfm_dir / "features.h5"
    pre_global = pre_sfm_dir / "global-descriptors.h5"
    pre_model = pre_sfm_dir / "sfm" / "reconstruction"  # Fixed path
    
    print("\n=== Step 1: Extract features from post-change images ===")
    feature_conf = {
        'output': 'feats-superpoint',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 8192,
            'keypoint_threshold': 0.005
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        }
    }
    
    extract_features.main(
        conf=feature_conf,
        image_dir=post_image_dir,
        export_dir=post_outputs,
        feature_path=post_features
    )
    
    print("\n=== Step 2: Extract global descriptors ===")
    global_conf = extract_features.confs["netvlad"]
    extract_features.main(
        conf=global_conf,
        image_dir=post_image_dir,
        export_dir=post_outputs,
        feature_path=post_global
    )
    
    print("\n=== Step 3: Retrieve similar pre-change images ===")
    pairs_from_retrieval.main(
        descriptors=post_global,
        output=pairs_file,
        num_matched=num_retrieval,
        db_descriptors=pre_global,
        db_model=pre_model
    )
    
    print("\n=== Step 4: Match features ===")
    matcher_conf = {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
            'match_threshold': 0.2,
        }
    }
    
    match_features.main(
        conf=matcher_conf,
        pairs=pairs_file,
        features=post_features,
        export_dir=post_outputs,
        matches=post_matches,
        features_ref=pre_features
    )
    
    print("\n=== Step 5: Localize post-change images ===")
    
    # Load pre-change reconstruction
    reconstruction = pycolmap.Reconstruction(str(pre_model))
    
    # Get camera from reconstruction
    camera = list(reconstruction.cameras.values())[0]
    
    # Setup localizer
    conf = {
        "estimation": {"ransac": {"max_error": ransac_thresh}},
        "refinement": {"refine_focal_length": False, "refine_extra_params": False},
    }
    localizer = QueryLocalizer(reconstruction, conf)
    
    # Get post-change image names
    post_images = sorted([p.name for p in post_image_dir.iterdir() 
                         if p.suffix in ['.jpg', '.png', '.jpeg']])
    
    # Load matches from h5 file
    import h5py
    matches_data = {}
    
    print("Loading matches from h5 file...")
    with h5py.File(post_matches, 'r') as f:
        # Each query image is a top-level group
        for query_name in f.keys():
            matches_data[query_name] = {}
            query_group = f[query_name]
            
            # Each reference image is a subgroup
            for ref_name in query_group.keys():
                matches_data[query_name][ref_name] = query_group[ref_name]['matches0'][()]
    
    print(f"Loaded matches for {len(matches_data)} query images")
    if len(matches_data) > 0:
        first_query = list(matches_data.keys())[0]
        print(f"Example: {first_query} has matches with {len(matches_data[first_query])} reference images")
    
    # Build name to ID mapping
    db_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}
    
    # Localize each image
    poses = {}
    failed = []
    
    for query_name in post_images:
        print(f"Localizing {query_name}...")
        
        if query_name not in matches_data:
            failed.append(query_name)
            print(f"  ✗ No matches found")
            continue
        
        # Get reference images for this query
        ref_names = list(matches_data[query_name].keys())
        ref_ids = [db_name_to_id[name] for name in ref_names if name in db_name_to_id]
        
        if not ref_ids:
            failed.append(query_name)
            print(f"  ✗ No valid reference images")
            continue
        
        try:
            # Prepare matches in the format expected by pose_from_cluster
            kpq = get_keypoints(post_features, query_name)
            kpq += 0.5  # COLMAP coordinates
            
            kp_idx_to_3D = defaultdict(list)
            kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
            num_matches = 0
            
            for i, db_id in enumerate(ref_ids):
                image = reconstruction.images[db_id]
                if image.num_points3D() == 0:
                    continue
                
                points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                         for p in image.points2D])
                
                # Get matches for this reference image
                ref_name = image.name
                if ref_name not in matches_data[query_name]:
                    continue
                    
                matches = matches_data[query_name][ref_name]
                
                # Convert matches to pairs of indices
                valid_matches = []
                for idx_q, idx_r in enumerate(matches):
                    if idx_r >= 0 and idx_r < len(points3D_ids):
                        if points3D_ids[idx_r] != -1:
                            valid_matches.append([idx_q, idx_r])
                
                if not valid_matches:
                    continue
                    
                valid_matches = np.array(valid_matches)
                num_matches += len(valid_matches)
                
                for idx_q, idx_r in valid_matches:
                    id_3D = points3D_ids[idx_r]
                    kp_idx_to_3D_to_db[idx_q][id_3D].append(i)
                    if id_3D not in kp_idx_to_3D[idx_q]:
                        kp_idx_to_3D[idx_q].append(id_3D)
            
            if num_matches == 0:
                failed.append(query_name)
                print(f"  ✗ No valid 2D-3D correspondences")
                continue
            
            # Prepare data for localization
            idxs = list(kp_idx_to_3D.keys())
            mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
            mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
            
            ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, camera)
            ret['camera'] = {
                'model': camera.model_name,
                'width': camera.width,
                'height': camera.height,
                'params': camera.params,
            }
            
            if ret['success']:
                qvec = ret['qvec']
                tvec = ret['tvec']
                
                # Convert to transform matrix (w2c)
                R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec
                
                # Convert to c2w (camera to world)
                c2w = np.linalg.inv(w2c)
                
                # Apply coordinate system fix (OpenCV to OpenGL)
                fix_rot = np.diag([1, -1, -1, 1])
                c2w = c2w @ fix_rot
                
                poses[query_name] = c2w
                
                print(f"  ✓ Success: {ret['num_inliers']} inliers")
            else:
                failed.append(query_name)
                print(f"  ✗ Failed to localize")
                
        except Exception as e:
            failed.append(query_name)
            print(f"  ✗ Error: {e}")
    
    print(f"\n=== Results ===")
    print(f"Successfully localized: {len(poses)}/{len(post_images)} images")
    if failed:
        print(f"Failed images: {failed}")
    
    # Save to transforms.json
    if poses:
        save_transforms_json(
            poses,
            post_transforms_file,
            camera
        )
        print(f"\n✓ Saved transforms to: {post_transforms_file}")
    
    return poses


def save_transforms_json(poses, output_path, camera):
    """Save poses to transforms.json format"""
    
    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": camera.params[0],
        "fl_y": camera.params[1],
        "cx": camera.params[2],
        "cy": camera.params[3],
        "w": camera.width,
        "h": camera.height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }
    
    for img_name, c2w in poses.items():
        frame = {
            "file_path": f"frames/{img_name}",
            "transform_matrix": c2w.tolist()
        }
        transforms["frames"].append(frame)
    
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)


if __name__ == "__main__":
    # Example usage
    pre_sfm_dir = Path("data_real/pre_static/pre_static_1_post/hloc_outputs")
    post_image_dir = Path("data_real/sync/prismatic/static_sync/frames")
    post_transforms_file = Path("data_real/sync/prismatic/static_sync/transforms.json")
    
    poses = localize_post_images(
        pre_sfm_dir=pre_sfm_dir,
        post_image_dir=post_image_dir,
        post_transforms_file=post_transforms_file,
        num_retrieval=10,
        ransac_thresh=12.0
    )