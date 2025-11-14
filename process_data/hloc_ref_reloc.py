import numpy as np
import json
from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.io import get_keypoints
from collections import defaultdict
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
pre_scene_dir = Path('data_real/day8/pre')
post_scene_dir = Path('data_real/day8/post')

# Pre-change reference model (metric scale preserved!)
pre_colmap_model = pre_scene_dir / 'colmap_ba'
pre_transforms = pre_scene_dir / 'transforms_colmap_metric.json'
pre_features = pre_scene_dir / 'hloc_outputs/features.h5'
pre_global_desc = pre_scene_dir / 'hloc_outputs/global-descriptors.h5'

# Post-change query images
post_images_dir = post_scene_dir / 'frames'
post_outputs = post_scene_dir / 'hloc_outputs'
post_outputs.mkdir(exist_ok=True, parents=True)

post_features = post_outputs / 'features.h5'
post_global_desc = post_outputs / 'global-descriptors.h5'
post_matches = post_outputs / 'matches.h5'

# ============================================================================
# STEP 1: Extract features from post-change images
# ============================================================================
print("="*80)
print("STEP 1: Extracting features from post-change images")
print("="*80)

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

post_image_list = [p.relative_to(post_images_dir).as_posix() 
                   for p in post_images_dir.iterdir() 
                   if p.suffix in ['.jpg', '.png']]

print(f"Extracting local features for {len(post_image_list)} post images...")
extract_features.main(
    feature_conf,
    post_images_dir,
    image_list=post_image_list,
    feature_path=post_features
)

print("Extracting global descriptors...")
global_conf = extract_features.confs["netvlad"]
extract_features.main(
    conf=global_conf,
    image_dir=post_images_dir,
    export_dir=post_outputs,
    feature_path=post_global_desc
)

# ============================================================================
# STEP 2: Image retrieval (post -> pre)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Retrieving similar pre-change images for each post image")
print("="*80)

def image_retrieval(post_desc, pre_desc, pre_model, num_matched=10):
    """Match post images to pre images using global descriptors"""
    from hloc import pairs_from_retrieval
    
    # Load pre-change image names
    pre_reconstruction = pycolmap.Reconstruction(str(pre_model))
    pre_names = [img.name for img in pre_reconstruction.images.values()]
    
    # Load descriptors
    post_names = pairs_from_retrieval.list_h5_names(post_desc)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    name2db = {n: 0 for n in pairs_from_retrieval.list_h5_names(pre_desc)}
    pre_desc_tensors = pairs_from_retrieval.get_descriptors(pre_names, [pre_desc], name2db)
    post_desc_tensors = pairs_from_retrieval.get_descriptors(post_names, post_desc)
    
    # Compute similarity
    sim = torch.einsum('id,jd->ij', post_desc_tensors.to(device), pre_desc_tensors.to(device))
    
    # Get top matches
    pairs = pairs_from_retrieval.pairs_from_score_matrix(
        sim, np.zeros((len(post_names), len(pre_names)), bool),
        num_matched, min_score=0
    )
    
    # Convert to dict
    pairs_dict = defaultdict(list)
    for i, j in pairs:
        pairs_dict[post_names[i]].append(pre_names[j])
    
    return pairs_dict

retrieval_pairs = image_retrieval(
    post_global_desc, pre_global_desc, pre_colmap_model, num_matched=10
)

print(f"Retrieved {len(retrieval_pairs)} post-pre image pairs")

# ============================================================================
# STEP 3: Feature matching (post -> pre)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Matching features between post and pre images")
print("="*80)

# Create pairs file
pairs_file = post_outputs / 'pairs-localization.txt'
with open(pairs_file, 'w') as f:
    for post_img, pre_imgs in retrieval_pairs.items():
        for pre_img in pre_imgs:
            f.write(f'{post_img} {pre_img}\n')

matcher_conf = {
    'output': 'matches-superglue',
    'model': {
        'name': 'superglue',
        'weights': 'outdoor',
        'sinkhorn_iterations': 50,
        'match_threshold': 0.2,
    }
}

# Match post features against pre features
match_features.main(
    matcher_conf,
    pairs_file,
    features=post_features,
    features_ref=pre_features,  # Reference features from pre-change
    export_dir=post_outputs,
    matches=post_matches
)

# Replace STEP 4 with this debugging version:

# ============================================================================
# STEP 4: Camera localization (PnP + RANSAC) - DEBUGGED VERSION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Localizing post-change cameras in pre-change coordinate frame")
print("="*80)

from hloc.localize_sfm import QueryLocalizer
import h5py

# Load pre-change transforms for camera intrinsics
with open(pre_transforms, 'r') as f:
    pre_data = json.load(f)

fx, fy = pre_data['fl_x'], pre_data['fl_y']
cx, cy = pre_data['cx'], pre_data['cy']
w, h = pre_data['w'], pre_data['h']

# Create camera model
camera = pycolmap.Camera(
    model='PINHOLE',
    width=int(w),
    height=int(h),
    params=[fx, fy, cx, cy]
)

# Load pre-change reconstruction
pre_reconstruction = pycolmap.Reconstruction(str(pre_colmap_model))
pre_name_to_id = {img.name: img_id for img_id, img in pre_reconstruction.images.items()}

print(f"Pre-change reconstruction has {len(pre_reconstruction.images)} images")
print(f"Pre-change reconstruction has {len(pre_reconstruction.points3D)} 3D points")

# Setup localizer with more lenient RANSAC
localization_config = {
    "estimation": {"ransac": {"max_error": 2}},  # Increased threshold
    "refinement": {
        'refine_focal_length': False
        # 'refine_extra_params': False,
        # 'refine_principal_point': False
    }
}
localizer = QueryLocalizer(pre_reconstruction, localization_config)

# Load matches
matches_h5 = h5py.File(post_matches, 'r')
print(f"Matches file contains {len(matches_h5.keys())} pairs")

# Debug: check a few match pairs
print("\nChecking first few match pairs:")
for i, key in enumerate(list(matches_h5.keys())[:3]):
    group = matches_h5[key]
    if 'matches0' in group:
        matches = group['matches0'][:]
    elif 'matches' in group:
        matches = group['matches'][:]
    else:
        matches = None
    if matches is not None:
        print(f"  {key}: {len(matches)} matches")
    else:
        print(f"  {key}: (no matches dataset found)")


def localize_image(query_name, retrieval_pairs, matches_h5):
    """Localize a single post-change image"""
    kpq = get_keypoints(post_features, query_name)
    kpq += 0.5  # COLMAP coordinates
    
    # Get retrieved pre-change images
    if query_name not in retrieval_pairs:
        return None, "No retrieval pairs"
    
    pre_images = retrieval_pairs[query_name]
    db_ids = [pre_name_to_id[name] for name in pre_images if name in pre_name_to_id]
    
    if len(db_ids) == 0:
        return None, "No valid db_ids"
    
    # Collect 2D-3D correspondences
    kp_idx_to_3D = defaultdict(list)
    total_matches = 0
    valid_matches = 0
    
    for db_id in db_ids:
        image = pre_reconstruction.images[db_id]
        if image.num_points3D() == 0:
            continue
        
        # Get 3D point IDs for this image
        points3D_ids = np.array([
            p.point3D_id if p.has_point3D() else -1
            for p in image.points2D
        ])
        
        # Try both naming conventions for matches
        pair_name1 = f'{query_name}/{image.name}'
        pair_name2 = f'{query_name}_{image.name}'
        pair_name3 = names_to_pair(query_name, image.name)  # hloc convention
        
        matches = None
        for pname in [pair_name1, pair_name2, pair_name3]:
            if pname in matches_h5:
                matches = matches_h5[pname]['matches0'][:]  # Try matches0 first
                if len(matches) == 0:
                    matches = matches_h5[pname][:]  # Fallback
                total_matches += len(matches)
                break
        
        if matches is None or len(matches) == 0:
            continue
        
        # Filter matches with valid 3D points
        # matches format: [query_idx, ref_idx] or just query_idx with -1 for no match
        if matches.ndim == 2:  # [N, 2] format
            valid_mask = points3D_ids[matches[:, 1]] != -1
            matches = matches[valid_mask]
        else:  # [N] format where value is ref_idx or -1
            valid_indices = np.where(matches != -1)[0]
            matches = np.stack([valid_indices, matches[valid_indices]], axis=1)
            valid_mask = points3D_ids[matches[:, 1]] != -1
            matches = matches[valid_mask]
        
        valid_matches += len(matches)
        
        for idx, m in matches:
            id_3D = points3D_ids[m]
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)
    
    if len(kp_idx_to_3D) < 4:  # Need at least 4 points for PnP
        return None, f"Insufficient 2D-3D: {len(kp_idx_to_3D)} (matches: {total_matches}, valid: {valid_matches})"
    
    # Prepare data for PnP
    mkp_idxs = [i for i in kp_idx_to_3D.keys() for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in kp_idx_to_3D.keys() for j in kp_idx_to_3D[i]]
    
    # Localize
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, camera)
    
    if not ret['success']:
        return None, f"PnP failed with {len(kp_idx_to_3D)} 2D-3D correspondences"
    
    return ret, f"Success: {ret['num_inliers']}/{len(mkp_idxs)} inliers"

# Helper function for hloc pair naming
def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

# Localize all post images with detailed logging
post_poses = {}
localized_count = 0
failure_reasons = defaultdict(int)

print("\nLocalizing images (showing first 5 failures):")
failure_count = 0

for query_name in tqdm(post_image_list, desc="Localizing"):
    ret, msg = localize_image(query_name, retrieval_pairs, matches_h5)
    
    if ret is not None and ret.get('success', False):
        post_poses[query_name] = {
            'qvec': ret['qvec'],
            'tvec': ret['tvec'],
            'num_inliers': ret['num_inliers']
        }
        localized_count += 1
    else:
        failure_reasons[msg] += 1
        if failure_count < 5:
            print(f"  Failed: {query_name} - {msg}")
            failure_count += 1

matches_h5.close()

print(f"\n✓ Successfully localized {localized_count}/{len(post_image_list)} post images")
print("\nFailure breakdown:")
for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]:
    print(f"  {reason}: {count} images")

# ============================================================================
# STEP 5: Convert to transforms.json format
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Saving post-change poses in pre-change coordinate frame")
print("="*80)

def colmap_to_transform_matrix(qvec, tvec):
    """Convert COLMAP pose to transform matrix"""
    w2c = np.eye(4)
    w2c[:3, :3] = pycolmap.qvec_to_rotmat(qvec)
    w2c[:3, 3] = tvec
    
    c2w = np.linalg.inv(w2c)
    
    # Convert OpenCV to OpenGL convention
    c2w[0:3, 1:3] *= -1
    
    return c2w

# Create transforms.json for post-change in pre-change frame
post_frames = []
for img_name in sorted(post_poses.keys()):
    pose_data = post_poses[img_name]
    c2w = colmap_to_transform_matrix(pose_data['qvec'], pose_data['tvec'])
    
    post_frames.append({
        'file_path': f'./frames/{img_name}',
        "depth_file_path": f"depth/{Path(img_name).stem}.npy",
        'transform_matrix': c2w.tolist(),
        'num_inliers': int(pose_data['num_inliers'])
    })

post_transforms = {
    'fl_x': fx,
    'fl_y': fy,
    'cx': cx,
    'cy': cy,
    'w': w,
    'h': h,
    'camera_angle_x': 2 * np.arctan(w / (2 * fx)),
    'camera_angle_y': 2 * np.arctan(h / (2 * fy)),
    'frames': post_frames
}

# Save
output_transforms_path = post_scene_dir / 'transforms_in_pre_frame.json'
with open(output_transforms_path, 'w') as f:
    json.dump(post_transforms, f, indent=2)

print(f"✓ Post-change transforms saved to: {output_transforms_path}")
print(f"\nKey achievement:")
print(f"  ✓ Post-change images localized in pre-change coordinate frame")
print(f"  ✓ Metric scale consistency preserved")
print(f"  ✓ Ready for direct pre-post comparison in 3DGS")