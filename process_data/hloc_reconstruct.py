"""
3D Reconstruction with hloc using NeRF-format data
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Reconstruction with hloc using NeRF-format data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the NeRF-format data directory (contains frames/, depth/, and transforms_arkit.json)"
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="transforms_arkit.json",
        help="Transform JSON file name (default: transforms_arkit.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hloc_outputs",
        help="Directory name for outputs (default: hloc_outputs)"
    )
    return parser.parse_args()


args = parse_args()
data_dir = Path(args.data_dir)
outputs = data_dir / args.output_dir
outputs.mkdir(exist_ok=True, parents=True)

# Derived paths
images = data_dir / "frames"
depth_dir = data_dir / "depth"
transforms_file = data_dir / args.transforms

# Output paths
feature_path = outputs / "features.h5"
matches_path = outputs / "matches.h5"
global_path = outputs / "global-descriptors.h5"
pairs_file = outputs / "pairs-retrieval.txt"
sfm_dir = outputs / "sfm"
sfm_dir.mkdir(exist_ok=True, parents=True)


# ============================================
# STEP 1: Extract Features
# ============================================
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


# ============================================
# STEP 2: Extract Global Descriptors
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
# STEP 5: Create Reference Model from transforms_arkit.json
# ============================================
print("\n=== Step 5: Creating reference model from transforms_arkit.json ===")

def create_reference_model_from_transforms(transforms_file, image_dir):
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
    
    temp_dir = Path("temp_colmap_model")
    temp_dir.mkdir(exist_ok=True)
    
    # Write cameras.txt
    with open(temp_dir / "cameras.txt", 'w') as f:
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt
    with open(temp_dir / "images.txt", 'w') as f:
        for idx, frame in enumerate(data['frames'], start=1):
            img_path = Path(frame['file_path'])
            img_name = img_path.name
            full_img_path = image_dir / img_name
            if not full_img_path.exists():
                print(f"Warning: Image not found: {full_img_path}")
                continue
            
            c2w = np.array(frame['transform_matrix'])
            fix_rot = np.diag([1, -1, -1, 1])
            w2c = np.linalg.inv(c2w @ fix_rot)
            
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            quat = Rotation.from_matrix(R).as_quat()
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{idx} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n\n")
    
    with open(temp_dir / "points3D.txt", 'w') as f:
        pass
    
    reconstruction = pycolmap.Reconstruction(str(temp_dir))
    print(f"Created reference model with {len(reconstruction.images)} images")
    
    shutil.rmtree(temp_dir)
    return reconstruction


reference_model = create_reference_model_from_transforms(transforms_file, images)
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
# STEP 7: Visualization
# ============================================
if model is not None:
    print("\n=== Step 7: Visualization ===")
    print(model.summary())
    
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
        scene=dict(aspectmode='data'),
        title="3D Reconstruction with hloc"
    )
    fig.write_html("reconstruction_viz.html", auto_open=True)
    
    output_path = sfm_dir / "reconstruction"
    output_path.mkdir(exist_ok=True, parents=True)
    model.write(str(output_path))
    print(f"✓ Reconstruction saved to: {output_path}")
    model.export_PLY(str(data_dir / "sparse_pc.ply"))
    print(f"✓ Sparse point cloud exported to: {data_dir / 'sparse_pc.ply'}")
else:
    print("\n✗ Reconstruction failed!")