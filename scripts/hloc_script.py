"""
3D Reconstruction with hloc using NeRF-format data
"""

from pathlib import Path
import shutil
import json
import cv2
import numpy as np
import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    triangulation,
    visualization,
    reconstruction
)
from hloc.utils import viz_3d


data_dir = Path("data_real/pre_static/pre_static_1_post")  # Your data directory
outputs = data_dir / "hloc_outputs"
outputs.mkdir(exist_ok=True, parents=True)

# Your data structure
images = data_dir / "frames"  # Changed from "images" to "frames"
depth_dir = data_dir / "depth"
transforms_file = data_dir / "transforms_pre.json"

# Output paths
feature_path = outputs / "features.h5"
matches_path = outputs / "matches.h5"
global_path = outputs / "global-descriptors.h5"
pairs_file = outputs / "pairs-retrieval.txt"
sfm_dir = outputs / "sfm"
sfm_dir.mkdir(exist_ok=True, parents=True)

# STEP 1: Extract Features
print("\n=== Step 1: Extracting features ===")
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

features = extract_features.main(
    conf=feature_conf,
    image_dir=images,
    export_dir=outputs,
    feature_path=feature_path
)

# STEP 2: Extract Global Descriptors for Retrieval
# ============================================
print("\n=== Step 2: Extracting global descriptors ===")
global_conf = extract_features.confs["netvlad"]
global_descriptors = extract_features.main(
    conf=global_conf,
    image_dir=images,
    export_dir=outputs,
    feature_path=global_path
)

# ============================================
# STEP 3: Generate Image Pairs
# ============================================
print("\n=== Step 3: Generating image pairs ===")
pairs_from_retrieval.main(
    descriptors=global_path,
    output=pairs_file,
    num_matched=10
)

print(f"Generated pairs saved to: {pairs_file}")

# ============================================
# STEP 4: Match Features
# ============================================
print("\n=== Step 4: Matching features ===")
matcher_conf = {
    'output': 'matches-superglue',
    'model': {
        'name': 'superglue',
        'weights': 'outdoor',
        'sinkhorn_iterations': 50,
        'match_threshold': 0.2,
    }
}

matches = match_features.main(
    conf=matcher_conf,
    pairs=pairs_file,
    features=feature_path,
    export_dir=outputs,
    matches=matches_path
)

# ============================================
# STEP 5: Create Reference Model from transforms_pre.json
# ============================================
print("\n=== Step 5: Creating reference model from transforms_pre.json ===")

def create_reference_model_from_transforms(transforms_file, image_dir):
    """Create COLMAP reference model from NeRF transforms.json"""
    from scipy.spatial.transform import Rotation
    
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    # Get camera parameters
    w = int(data['w'])
    h = int(data['h'])
    fx = float(data['fl_x'])
    fy = float(data['fl_y'])
    cx = float(data['cx'])
    cy = float(data['cy'])
    
    print(f"Camera parameters: {w}x{h}, fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Create temporary directory for COLMAP text files
    temp_dir = Path("temp_colmap_model")
    temp_dir.mkdir(exist_ok=True)
    
    # Write cameras.txt
    with open(temp_dir / "cameras.txt", 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt
    with open(temp_dir / "images.txt", 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for idx, frame in enumerate(data['frames'], start=1):
            # Get image path
            img_path = Path(frame['file_path'])
            img_name = img_path.name
            
            # Check if image exists
            full_img_path = image_dir / img_name
            if not full_img_path.exists():
                print(f"Warning: Image not found: {full_img_path}")
                continue
            
            # Get transform matrix (c2w - camera to world)
            c2w = np.array(frame['transform_matrix'])
            
            fix_rot = np.diag([1, -1, -1, 1])
            w2c = np.linalg.inv(c2w @ fix_rot)

            
            # Extract rotation and translation
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            quat = Rotation.from_matrix(R).as_quat()  # Returns [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]  # Reorder to [w, x, y, z]
            
            # Write image line
            f.write(f"{idx} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n")
            f.write("\n")  # Empty line for POINTS2D
    
    # Write empty points3D.txt
    with open(temp_dir / "points3D.txt", 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    # Read back as COLMAP reconstruction using pycolmap 0.4.0 API
    reconstruction = pycolmap.Reconstruction(str(temp_dir))
    
    print(f"Created reference model with {len(reconstruction.images)} images")
    
    # Clean up temp files
    shutil.rmtree(temp_dir)
    
    return reconstruction   

reference_model = create_reference_model_from_transforms(transforms_file, images)

# Save reference model
ref_model_path = outputs / "reference_model"
ref_model_path.mkdir(exist_ok=True)
reference_model.write_text(str(ref_model_path))
print(f"Reference model saved to: {ref_model_path}")

# ============================================
# STEP 6: Triangulation with Known Poses
# ============================================
print("\n=== Step 6: Triangulating 3D points ===")

model = triangulation.main(
    sfm_dir=sfm_dir,
    reference_model=ref_model_path,
    image_dir=images,
    pairs=pairs_file,
    features=feature_path,
    matches=matches_path,
    skip_geometric_verification=True
)

# ============================================
# ALTERNATIVE: Full SfM (Estimate Poses from Scratch)
# ============================================
# Uncomment this if you want to estimate poses instead of using transforms_pre.json
# print("\n=== Alternative: Running full SfM reconstruction ===")
# model = reconstruction.main(
#     sfm_dir=sfm_dir,
#     image_dir=images,
#     pairs=pairs_file,
#     features=feature_path,
#     matches=matches_path
# )

# ============================================
# STEP 7: Visualize Results
# ============================================
if model is not None:
    print("\n=== Step 7: Visualization ===")
    print(model.summary())
    
    # 3D visualization
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        fig,
        model,
        color="rgba(255,0,0,0.5)",
        name="reconstruction",
        points=len(model.points3D) > 0,
        cameras=True,
        cs=0.1,
        points_rgb=True
    )
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        title="3D Reconstruction with hloc"
    )
    fig.write_html("reconstruction_viz.html", auto_open=True)

    
    # 2D visualization
    print("\n=== Feature match visualization ===")
    visualization.visualize_sfm_2d(
        model,
        images,
        color_by="visibility",
        n=2
    )
    
    # Save final model
    output_path = sfm_dir / "reconstruction"
    output_path.mkdir(exist_ok=True, parents=True)
    model.write(str(output_path))
    print(f"\n✓ Reconstruction saved to: {output_path}")
    print(f"✓ Number of 3D points: {len(model.points3D)}")
    print(f"✓ Number of registered images: {len(model.images)}")


    model.export_PLY(str(data_dir / "sparse_pc.ply"))
    print(f"✓ Sparse point cloud exported to: {data_dir / 'sparse_pc.ply'}")
else:
    print("\n✗ Reconstruction failed!")

# ============================================
# STEP 8 (Optional): Localize a Query Image
# ============================================
"""
# Example: Localize a specific frame
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

query_name = "frame_00100.jpg"  # Change to your query image
query_path = images / query_name

if query_path.exists() and model is not None:
    print(f"\n=== Localizing query image: {query_name} ===")
    
    # Setup localizer
    camera = list(model.cameras.values())[0]
    ref_ids = list(model.images.keys())
    
    conf = {
        "estimation": {"ransac": {"max_error": 12}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }
    
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(
        localizer,
        query_name,
        camera,
        ref_ids,
        feature_path,
        matches_path
    )
    
    if ret is not None:
        print(f'✓ Found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences')
        
        # Visualize
        visualization.visualize_loc_from_log(images, query_name, log, model)
"""