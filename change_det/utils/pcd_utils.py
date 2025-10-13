# Point cloud processing functions for for NeRFacto2
import cv2
import numpy as np
import open3d as o3d
import torch

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def compute_point_cloud(depths, poses, Ks, masks):
    """
    Compute the point cloud from depth maps, camera poses and intrinsics
    Args:
        depths (N, 1, H, W): Depth maps
        poses (N, 4, 4): Camera poses
        Ks (N, 3, 3): Camera intrinsics
        masks (N, 1, H, W): Binary masks
    Returns:
        point_cloud (torch.Tensor): Point cloud
    """
    # Handle possible extra dimension (e.g., [N,1,1,H,W])
    if depths.ndim == 5 and depths.shape[2] == 1:
        depths = depths.squeeze(2)
    elif depths.ndim != 4:
        raise ValueError(f"Expected depths to have shape (N,1,H,W), got {depths.shape}")

    # Ensure all tensors are on same device
    device = depths.device
    poses = poses.to(device)
    Ks = Ks.to(device)
    masks = masks.to(device)

    N, _, H, W = depths.shape

    # Create pixel coordinate grid
    u = torch.linspace(0, W - 1, W, device=device).repeat(H, 1).expand(N, H, W)
    v = torch.linspace(0, H - 1, H, device=device).repeat(W, 1).t().expand(N, H, W)

    # Compute normalized camera coordinates
    X = (u - Ks[:, 0, 2].view(N, 1, 1)) / Ks[:, 0, 0].view(N, 1, 1) * depths.squeeze(1)
    Y = (v - Ks[:, 1, 2].view(N, 1, 1)) / Ks[:, 1, 1].view(N, 1, 1) * depths.squeeze(1)
    Z = depths.squeeze(1)

    # Stack points in camera coordinates
    point_cloud = torch.stack((X, Y, Z), dim=-1)

    # Convert to homogeneous coordinates
    ones = torch.ones(N, H, W, 1, device=device)
    point_cloud_hom = torch.cat((point_cloud, ones), dim=-1)

    # Transform points to world coordinates
    point_cloud_world = torch.einsum('bij,bklj->bkli', poses, point_cloud_hom)

    # Reshape and mask
    point_cloud_world_reshaped = point_cloud_world.reshape(-1, 4)
    non_zero_mask = masks.view(-1).bool()
    point_cloud_in_mask = point_cloud_world_reshaped[non_zero_mask, :3]

    return point_cloud_in_mask



def point_cloud_filtering(point_cloud, percentile_threshold=0.90):
    """
    Simple percentile-based point cloud outlier filtering
    TODO: Improve the outlier filtering strategy
    
    Args:
        point_cloud (Nx3): input point cloud
    
    Returns:
        point_cloud_inliers (Mx3): inlier point cloud
    """
    # Compute the centroid of the point cloud
    centroid = torch.median(point_cloud, dim=-2)[0]
    # Compute distances of all points to the centroid
    distances = torch.norm(point_cloud - centroid, dim=-1)
    # Filter out the paints that are beyond the 95% percentile
    threshold_distance = torch.quantile(distances, percentile_threshold)
    inlier_mask = distances < threshold_distance
    point_cloud_inliers = point_cloud[inlier_mask]
    return point_cloud_inliers

def mahalanobis_filter(point_cloud, percentile_threshold=0.90):
    mean = torch.mean(point_cloud, dim=0)
    cov = torch.cov(point_cloud.T)
    cov_inv = torch.inverse(cov + 1e-6 * torch.eye(3, device=point_cloud.device)) 

    diffs = point_cloud - mean
    dists = torch.sqrt(torch.sum(diffs @ cov_inv * diffs, dim=1))
    threshold = torch.quantile(dists, percentile_threshold)
    inliers = dists < threshold
    return point_cloud[inliers]



def nn_distance(point_cloud1, point_cloud2):
    """
    Efficiently compute the approximate NN distance between two point clouds

    Args:
        point_cloud1 (N, 3): First point cloud
        point_cloud2 (M, 3): Second point cloud
    
    Returns:
        distances (float): Nearest neighbor distance
    """
    assert point_cloud1.shape[-1] == 3
    assert point_cloud2.shape[-1] == 3
    # import time
    # start = time.time()
    index_params = dict(algorithm = 1, trees = 1)  # KDTree
    search_params = dict(checks = 2) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([point_cloud1.cpu().numpy()])
    flann.train()
    # Find the nearest neighbors
    matches = flann.match(point_cloud2.cpu().numpy())
    smallest_distance = min(match.distance for match in matches)
    # print(f"KDTree query time: {time.time() - start}")
    return smallest_distance


def pcd_size(pcd):
    """
    Efficiently compute the maximum distance between two points in a pcd

    Args:
        pcd (Nx3): point cloud

    Returns:
        max_dist (float): Maximum distance between two points in a pcd
    """
    # Max-distance points must lie on the convex hull
    hull = ConvexHull(pcd.cpu().numpy())
    hullpoints = pcd.cpu().numpy()[hull.vertices, :]
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    return hdist.max()


def FPS(pcd, num=100):
    """
    Farthest point sampling for a point cloud

    Args:
        pcd (Nx3): Point cloud
        num (int): Number of points to sample

    Returns:
        indices (M): Indices of the sampled points
        pcd_fps (Mx3): Sampled points
    """
    assert pcd.shape[-1] == 3
    if pcd.shape[0] < num:
        return torch.arange(pcd.shape[0]).to(pcd.device), pcd
    device, dtype = pcd.device, pcd.dtype
    pcd = pcd.cpu().numpy()
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd_fps = np.asarray(pcd_o3d.farthest_point_down_sample(num).points)
    indices = np.nonzero((pcd[:, None, :] == pcd_fps[None, :, :]).all(-1))[0]
    pcd_fps = torch.from_numpy(pcd_fps).to(device).to(dtype)
    indices = torch.tensor(indices).to(device)
    return indices, pcd_fps


def transform_point_cloud(point_cloud, transform_matrix): 
    """
    Transforms a point cloud using a given transformation matrix.
    
    Args:
        point_cloud (..., 3): point cloud.
        transform_matrix (4, 4): 4x4 transformation matrix.

    Returns:
        transformed_point_cloud (..., 3): Transformed point cloud of shape.
    """
    assert point_cloud.shape[-1] == 3
    assert transform_matrix.shape == (4, 4)
    # Get the shape of the input point cloud
    *leading_dims, _ = point_cloud.shape
    # Convert point cloud to homogeneous coordinates
    ones = torch.ones((*leading_dims, 1)).to(point_cloud)
    point_cloud_homo = torch.cat((point_cloud, ones), dim=-1)
    #Transform the point cloud using the transformation matrix
    transformed_point_cloud_homo = torch.einsum(
        "ij,...j->...i", transform_matrix, point_cloud_homo
    )
    # Extract the 3D coordinates of the transformed point cloud 
    transformed_point_cloud = transformed_point_cloud_homo[..., :3]
    return transformed_point_cloud


def compute_3D_bbox(point_cloud):
    """
    Computes the 3D bounding box
    from depth maps, camera pases and intrinsics.

    Args:
        point cloud (Nx3 Tensor): point cloud.

    Returns:
        bbox_min (3-Tuple): Minimum corner of the bounding box.
        bbox_max (3-Tuple): Maximum corner of the bounding box.
    """
    assert point_cloud.shape[-1] == 3
    bbox_min = torch.min(point_cloud, dim=0)[0].cpu().numpy()
    bbox_max = torch.max(point_cloud, dim=0)[0].cpu().numpy()
    return tuple(bbox_min), tuple(bbox_max)


def expand_3D_bbox(bbox3d, percentage=.8):
    """
    Expand 3D bounding box by a certain percentage

    Args:
        bbox3d ((bbox_min, bbox_max)): 3D bounding box
        percentage (float): Percentage to expand

    Return:
        bbox3d_new ((bbox_min, bbox_max)): Expanded 3D bounding box
    """
    bbox_min, bbox_max = bbox3d
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    # Calculate the center of the bounding box
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2
    # Calculate half-widths
    x_half = x_center - xmin
    y_half = y_center - ymin
    z_half = z_center - zmin
    # Calculate expansion amounts
    x_expand = x_half * percentage
    y_expand = y_half * percentage
    z_expand = z_half * percentage
    # Calculate new bounding box corners
    xmin_new = x_center - (x_half + x_expand)
    xmax_new = x_center + (x_half + x_expand)
    ymin_new = y_center - (y_half + y_expand)
    ymax_new = y_center + (y_half + y_expand)
    zmin_new = z_center - (z_half + z_expand)
    zmax_new = z_center + (z_half + z_expand)
    return (xmin_new, ymin_new, zmin_new), (xmax_new, ymax_new, zmax_new)


def bbox2voxel(bbox3d, grid_size=400, device="cuda"):
    """
    Initialize a voxel grid for a 3D bounding box

    Args:
        bbox3d ((3-tuple, 3-tuple)): Point cloud
        grid_size (int): Resolution of the voxel grid

    Returns:
        voxel (GxGxG): Voxel grid
    """
    assert len(bbox3d[0]) == len(bbox3d[1]) == 3
    xs = torch.linspace(bbox3d[0][0], bbox3d[1][0], grid_size)
    ys = torch.linspace(bbox3d[0][1], bbox3d[1][1], grid_size)
    zs = torch.linspace(bbox3d[0][2], bbox3d[1][2], grid_size)
    xs, ys, zs = torch.meshgrid(xs, ys, zs, indexing='ij')
    voxel = torch.stack((xs, ys, zs), dim=-1).to(device)
    return voxel


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualize_bbox3d_matplotlib(bbox):
    bbox_min, bbox_max = bbox

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max

    # Define the 8 corners of the box
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ])

    # Define 12 edges connecting the corners
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    lines = [(corners[start], corners[end]) for start, end in edges]
    ax.add_collection3d(Line3DCollection(lines, colors='r', linewidths=2))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("3D Bounding Box")
    plt.show()

def project_bbox3d_on_image(bbox_min, bbox_max, cam_pose, K, dist, image, save_path=None):
    # 8 corners of bbox
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]]
    ])  # shape: (8, 3)

    corners = torch.tensor(corners).float().to(image.device if torch.is_tensor(image) else "cpu")  # (8, 3)

    # Project to 2D
    proj_2d, valid = project_points(corners, cam_pose[None], K[None], dist[None], image.shape[-2], image.shape[-1])
    proj_2d = proj_2d.squeeze(1).cpu().numpy()

    image_np = image[0].permute(1, 2, 0).cpu().numpy() if torch.is_tensor(image) else image.copy()
    image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np

    # Draw edges
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    for (i, j) in edges:
        pt1, pt2 = tuple(proj_2d[i].astype(int)), tuple(proj_2d[j].astype(int))
        cv2.line(image_np, pt1, pt2, (0, 0, 255), 2)

    if save_path:
        cv2.imwrite(save_path, image_np)

    cv2.imshow("bbox projection", image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def points_to_occupancy(points, bbox_min, bbox_max, voxel_dim):
    """
    Convert a point cloud to a binary voxel grid.
    """
    device = points.device

    bbox_min = torch.tensor(bbox_min, dtype=points.dtype, device=device)
    bbox_max = torch.tensor(bbox_max, dtype=points.dtype, device=device)
    scales = torch.tensor(voxel_dim, dtype=points.dtype, device=device) / (bbox_max - bbox_min)

    voxel_coords = ((points - bbox_min) * scales).long()

    for i in range(3):
        voxel_coords[:, i] = voxel_coords[:, i].clamp(0, voxel_dim[i] - 1)

    voxel_grid = torch.zeros(*voxel_dim, dtype=torch.bool, device=device)
    voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

    return voxel_grid



