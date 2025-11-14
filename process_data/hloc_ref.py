import numpy as np
import json
import os
import pycolmap
import shutil
from pathlib import Path
from pyquaternion import Quaternion
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
from hloc.utils.read_write_model import Camera, Image, Point3D, write_model, read_model
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================
scene_dir = Path('data_real/day8/pre')  # or './post'
images_dir = scene_dir / 'frames'
transforms_arkit_path = scene_dir / 'transforms_arkit.json'
depth_dir = scene_dir / 'depth'  # .npy files

# Output directories
outputs = scene_dir / 'hloc_outputs'
outputs.mkdir(exist_ok=True, parents=True)

colmap_arkit_dir = scene_dir / 'colmap_arkit'
colmap_sparse_dir = scene_dir / 'colmap_sparse'
colmap_ba_dir = scene_dir / 'colmap_ba'

# HLOC paths
feature_path = outputs / 'features.h5'
matches_path = outputs / 'matches.h5'
pairs_file = outputs / 'pairs-sfm.txt'
global_path = outputs / 'global-descriptors.h5'

# Parameters
n_matched = 10
n_ba_iterations = 2  # Number of triangulation + BA iterations

# ============================================================================
# STEP 1: LOAD ARKIT DATA AND CREATE COLMAP REFERENCE MODEL
# ============================================================================
print("="*80)
print("STEP 1: Creating COLMAP reference model from ARKit poses")
print("="*80)

def convert_arkit_to_colmap_pose(c2w_arkit):
    """Convert ARKit camera-to-world to COLMAP world-to-camera format"""
    # ARKit uses different coordinate conventions
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w_cv = np.matmul(c2w_arkit, flip_yz)
    w2c_cv = np.linalg.inv(c2w_cv)
    return w2c_cv

# Load transforms_arkit.json
with open(transforms_arkit_path, 'r') as f:
    arkit_data = json.load(f)

# Extract camera intrinsics
fx = arkit_data['fl_x']
fy = arkit_data['fl_y']
cx = arkit_data['cx']
cy = arkit_data['cy']
w = arkit_data['w']
h = arkit_data['h']

# Create COLMAP model with ARKit poses
images_colmap = {}
cameras_colmap = {}
points3D = {}

# Create single shared camera (assuming all frames use same intrinsics)
camera_id = 1
camera = Camera(
    id=camera_id,
    model='PINHOLE',
    width=int(w),
    height=int(h),
    params=[fx, fy, cx, cy]
)
cameras_colmap[camera_id] = camera

# Process each frame
for idx, frame in enumerate(arkit_data['frames']):
    image_name = Path(frame['file_path']).name
    
    # Extract pose
    c2w_arkit = np.array(frame['transform_matrix'])
    w2c_cv = convert_arkit_to_colmap_pose(c2w_arkit)
    
    # Convert to quaternion + translation
    R = w2c_cv[:3, :3]
    q = Quaternion(matrix=R, atol=1e-06)
    qvec = np.array([q.w, q.x, q.y, q.z])
    tvec = w2c_cv[:3, -1]
    
    # Create COLMAP image
    image_id = idx + 1
    image = Image(
        id=image_id,
        qvec=qvec,
        tvec=tvec,
        camera_id=camera_id,
        name=image_name,
        xys=[],
        point3D_ids=[]
    )
    images_colmap[image_id] = image

# Write ARKit-based COLMAP model
colmap_arkit_dir.mkdir(exist_ok=True, parents=True)
write_model(
    images=images_colmap,
    cameras=cameras_colmap,
    points3D=points3D,
    path=str(colmap_arkit_dir),
    ext='.bin'
)
print(f"✓ ARKit reference model saved to: {colmap_arkit_dir}")
print(f"  - {len(images_colmap)} images")
print(f"  - {len(cameras_colmap)} cameras")

# ============================================================================
# STEP 2: FEATURE EXTRACTION AND MATCHING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Feature extraction and matching")
print("="*80)

# Feature extraction config
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

# Extract local features
print("\nExtracting local features...")
references = [str(p.relative_to(images_dir)) for p in images_dir.iterdir() if p.suffix in ['.jpg', '.png']]
extract_features.main(
    feature_conf,
    images_dir,
    image_list=references,
    feature_path=feature_path
)
print(f"✓ Features extracted: {feature_path}")

# Extract global descriptors for retrieval
print("\nExtracting global descriptors...")
global_conf = extract_features.confs["netvlad"]
extract_features.main(
    conf=global_conf,
    image_dir=images_dir,
    export_dir=outputs,
    feature_path=global_path
)
print(f"✓ Global descriptors extracted: {global_path}")

# Generate pairs from poses (better than retrieval for sequential data)
print("\nGenerating image pairs from poses...")
from hloc import pairs_from_poses
pairs_from_poses.main(colmap_arkit_dir, pairs_file, n_matched)
print(f"✓ Pairs generated: {pairs_file}")

# Match features
print("\nMatching features...")
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
    matcher_conf,
    pairs_file,
    features=feature_path,
    matches=matches_path
)
print(f"✓ Matches computed: {matches_path}")

# ============================================================================
# STEP 3: ITERATIVE TRIANGULATION + BUNDLE ADJUSTMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Iterative triangulation and bundle adjustment")
print("="*80)

colmap_input = colmap_arkit_dir

for iteration in range(n_ba_iterations):
    print(f"\n--- Iteration {iteration + 1}/{n_ba_iterations} ---")
    
    # Triangulation
    print("Running triangulation...")
    colmap_sparse_dir.mkdir(exist_ok=True, parents=True)
    
    reconstruction = triangulation.main(
        sfm_dir=colmap_sparse_dir,
        reference_model=colmap_input,
        image_dir=images_dir,
        pairs=pairs_file,
        features=feature_path,
        matches=matches_path,
        skip_geometric_verification=False
    )
    
    if reconstruction is None:
        print("⚠ Triangulation failed!")
        break
    
    print(f"✓ Triangulation complete:")
    print(f"  - {len(reconstruction.images)} images")
    print(f"  - {len(reconstruction.points3D)} 3D points")
    
    # Bundle Adjustment (preserve scale by not refining intrinsics)
    print("Running bundle adjustment...")
    colmap_ba_dir.mkdir(exist_ok=True, parents=True)
    
    ba_cmd = f'''colmap bundle_adjuster \
        --input_path {colmap_sparse_dir} \
        --output_path {colmap_ba_dir} \
        --BundleAdjustment.refine_focal_length 0 \
        --BundleAdjustment.refine_principal_point 0 \
        --BundleAdjustment.refine_extra_params 0 \
        --BundleAdjustment.max_num_iterations 50'''
    
    os.system(ba_cmd)
    print(f"✓ Bundle adjustment complete: {colmap_ba_dir}")
    
    # Update input for next iteration
    colmap_input = colmap_ba_dir
    
    # Clean up intermediate sparse dir
    if iteration < n_ba_iterations - 1:
        shutil.rmtree(colmap_sparse_dir)
        colmap_sparse_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 4: EXPORT REFINED POSES TO NERFSTUDIO FORMAT
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Exporting refined poses to Nerfstudio format")
print("="*80)

# Load final BA model
final_model = pycolmap.Reconstruction(str(colmap_ba_dir))

def convert_colmap_to_arkit_pose(w2c_cv):
    """Convert COLMAP world-to-camera back to ARKit camera-to-world"""
    c2w_cv = np.linalg.inv(w2c_cv)
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w_arkit = np.matmul(c2w_cv, flip_yz)
    return c2w_arkit

# Export refined transforms
frames_refined = []
for image_id in sorted(final_model.images.keys()):
    image = final_model.images[image_id]
    
    # Convert COLMAP pose to ARKit format
    w2c_cv = np.eye(4)
    w2c_cv[:3, :3] = pycolmap.qvec_to_rotmat(image.qvec)
    w2c_cv[:3, -1] = image.tvec
    c2w_arkit = convert_colmap_to_arkit_pose(w2c_cv)
    
    frames_refined.append({
        "file_path": f"frames/{image.name}",
        "depth_file_path": f"depth/{Path(image.name).stem}.npy",
        "transform_matrix": c2w_arkit.tolist()
    })

# Sort by filename
frames_refined = sorted(frames_refined, key=lambda x: x['file_path'])

# Create refined transforms
transforms_refined = {
    'fl_x': fx,
    'fl_y': fy,
    'cx': cx,
    'cy': cy,
    'w': w,
    'h': h,
    'camera_angle_x': 2 * np.arctan(w / (2 * fx)),
    'camera_angle_y': 2 * np.arctan(h / (2 * fy)),
    'ply_file_path': "fused_pc.ply",
    'frames': frames_refined
}

# Save refined transforms (METRIC SCALE PRESERVED!)
output_transforms_path = scene_dir / 'transforms_colmap_metric.json'
with open(output_transforms_path, 'w') as f:
    json.dump(transforms_refined, f, indent=2)

print(f"✓ Refined transforms saved to: {output_transforms_path}")
print(f"  - Metric scale preserved (ARKit initialization)")
print(f"  - Geometric accuracy improved (COLMAP refinement)")

# ============================================================================
# STEP 5: PREPARE FOR QED-SPLATTER
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Preparing data for QED-Splatter")
print("="*80)

# Create QED-Splatter compatible directory structure
qed_input_dir = scene_dir / 'qed_input'
qed_input_dir.mkdir(exist_ok=True)

# Copy images
qed_images_dir = qed_input_dir / 'images'
if qed_images_dir.exists():
    shutil.rmtree(qed_images_dir)
shutil.copytree(images_dir, qed_images_dir)

# Copy depth maps
qed_depth_dir = qed_input_dir / 'depth'
if qed_depth_dir.exists():
    shutil.rmtree(qed_depth_dir)
shutil.copytree(depth_dir, qed_depth_dir)

# Copy refined transforms
shutil.copy(output_transforms_path, qed_input_dir / 'transforms.json')

# Create metadata file for QED-Splatter
metadata = {
    'depth_unit_scale_factor': 1.0,  # Assuming depth is already in meters
    'orientation_method': 'none',
    'center_method': 'none',
    'auto_scale_poses': False,  # CRITICAL: Keep metric scale!
}

with open(qed_input_dir / 'dataparser_config.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ QED-Splatter input prepared: {qed_input_dir}")
print("\nTo train with QED-Splatter, run:")
print(f"  ns-train qed-splatter --data {qed_input_dir}")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nOutputs:")
print(f"  - ARKit reference model: {colmap_arkit_dir}")
print(f"  - Final refined model: {colmap_ba_dir}")
print(f"  - Metric transforms: {output_transforms_path}")
print(f"  - QED-Splatter input: {qed_input_dir}")
print(f"\nKey features:")
print(f"  ✓ Metric scale preserved from ARKit")
print(f"  ✓ Geometric accuracy from COLMAP refinement")
print(f"  ✓ Ready for depth-supervised 3DGS")