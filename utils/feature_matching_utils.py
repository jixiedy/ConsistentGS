import os
import re
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

import torch
import torch.nn.functional as F

from utils.graphics_utils import fov2focal



def extract_filename_without_extension(filename):
    """
    Get the filename without extension, supporting complex filenames.
    
    args:
    filename (str): Full filename, can contain any valid characters
    
    returns:
    str: Filename part without extension
    """
    # Define common image file extensions
    common_extensions = r'\.(jpg|jpeg|png|gif|bmp|webp|tiff|svg)$'
    
    # Use regex to match the last file extension
    pattern = f'(?i){common_extensions}'
    
    # Replace the last file extension
    result = re.sub(pattern, '', filename)
    
    # Handle both Unix and Windows path separators
    if '/' in result:
        result = '-'.join(result.split('/'))
    elif '\\' in result:
        result = '-'.join(result.split('\\'))
    
    return result


def sample_depths(depth_map, keypoints, sample_type="bilinear"):
    """
    Optimized depth sampling function
    
    Args:
        depth_map: torch.Tensor, depth map [H,W]
        keypoints: torch.Tensor, keypoint coordinates [N,2]
        sample_type: str, sampling method ["nearest", "bilinear"]
    Returns:
        torch.Tensor: Sampled depth values [N]
    """
    if sample_type == "nearest":
        keypoints_idx = keypoints.round().long()
        return depth_map[keypoints_idx[:, 1], keypoints_idx[:, 0]]
    
    # Calculate normalized coordinates in a single operation
    h, w = depth_map.shape
    keypoints_norm = 2.0 * torch.stack([
        keypoints[:, 0] / (w - 1) - 1,
        keypoints[:, 1] / (h - 1) - 1
    ], dim=-1)
    
    grid = keypoints_norm.view(1, -1, 1, 2)
    sampled = F.grid_sample(depth_map.unsqueeze(0).unsqueeze(0),
                           grid,
                           mode='bilinear',
                           padding_mode='border',
                           align_corners=True)
    
    return sampled.squeeze()


def process_depth_and_points(depth, keypoints, image_shape, camera, sample_type="bilinear"):
    """
    Process depth map and keypoints
    Args:
        depth: Depth map [1,H,W] or [H,W]
        keypoints: Keypoint coordinates [N,2]
        image_shape: Original image size (H,W)
        camera: Camera parameters
        sample_type: Depth sampling method
    Returns:
        depth: Processed depth map [H,W]
        points3d_world: 3D points in world coordinates [N,3]
        keypoints_scaled: Scaled keypoint coordinates [N,2]
        depths: Sampled depth values [N]
    """
    depth = depth.squeeze()
    
    # Calculate scaling factors
    scale_w = depth.shape[1] / image_shape[1]
    scale_h = depth.shape[0] / image_shape[0]
    
    # Scale keypoints
    keypoints_scaled = keypoints.clone()
    keypoints_scaled[:, 0] = (keypoints[:, 0] * scale_w).clamp(0, depth.shape[1] - 1)
    keypoints_scaled[:, 1] = (keypoints[:, 1] * scale_h).clamp(0, depth.shape[0] - 1)
    
    # Sample depth values
    depths = sample_depths(depth, keypoints_scaled, sample_type)
    
    # Calculate camera intrinsics
    fx = fov2focal(camera.FoVx, depth.shape[1])
    fy = fov2focal(camera.FoVy, depth.shape[0])
    cx = depth.shape[1] / 2
    cy = depth.shape[0] / 2
    
    # Back-project to camera coordinates  
    x = (keypoints_scaled[:, 0] - cx) * depths / fx
    y = (keypoints_scaled[:, 1] - cy) * depths / fy
    points3d = torch.stack([x, y, depths], dim=1)
    
    # Transform to world coordinates
    world_transform = camera.world_view_transform.transpose(0, 1).inverse()
    if not torch.is_tensor(world_transform):
        world_transform = torch.tensor(world_transform, device=camera.data_device, dtype=torch.float32)
    
    points3d_homo = torch.cat([points3d, torch.ones_like(points3d[:, :1])], dim=1)
    points3d_world = (world_transform @ points3d_homo.T).T[:, :3]
    
    return depth, points3d_world, keypoints_scaled, depths


def process_mono_depth(depth, keypoints, image_shape, camera, sample_type="bilinear"):
    """
    Process depth map and keypoints
    Args:
        depth: Depth map [1,H,W] or [H,W]
        keypoints: Keypoint coordinates [N,2]
        image_shape: Original image size (H,W)
        camera: Camera parameters
        sample_type: Depth sampling method
    Returns:
        depth: Processed depth map [H,W]
        depths: Sampled depth values [N]

    """
    # Depth map dimension processing
    depth = depth.squeeze()
    
    # Calculate scaling factors
    scale_w = depth.shape[1] / image_shape[1]
    scale_h = depth.shape[0] / image_shape[0]
    
    # Scale keypoints
    keypoints_scaled = keypoints.clone()
    keypoints_scaled[:, 0] = (keypoints[:, 0] * scale_w).clamp(0, depth.shape[1] - 1)
    keypoints_scaled[:, 1] = (keypoints[:, 1] * scale_h).clamp(0, depth.shape[0] - 1)
    
    # Sample depth values
    depths = sample_depths(depth, keypoints_scaled, sample_type)
    
    return depth, depths


def save_ply(filepath, points, colors):
    """Save colored point cloud as PLY file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    assert colors.max() <= 255 and colors.min() >= 0, "Colors should be in range [0,255]"
    assert len(points) == len(colors), "Points and colors must have same length"
    colors = colors.astype(np.uint8)

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")


def save_debug_pointcloud(validated_point_cloud_path, iteration, current_name, next_name, 
                         points0, points1, max_points=1000):
    """
    Save debug information for point clouds
    Args:
        validated_point_cloud_path: Save path
        iteration: Current iteration number
        current_name: Current frame image name
        next_name: Next frame image name
        points0: 3D points of the current frame
        points1: 3D points of the next frame
        max_points: Maximum number of points to save
    """

    os.makedirs(validated_point_cloud_path, exist_ok=True)
    
    # Limit number of sampled points
    n_points = len(points0)
    if n_points > max_points:
        indices = torch.randperm(n_points)[:max_points]
        points0 = points0[indices]
        points1 = points1[indices]
    
    # Generate random colors for each pair of points
    colors = torch.randint(0, 255, (len(points0), 3), device=points0.device)
    
    # Save point clouds for both frames
    current_image_name_prefix = extract_filename_without_extension(current_name)
    next_image_name_prefix = extract_filename_without_extension(next_name)
    save_ply(f"{validated_point_cloud_path}/{iteration}_{current_image_name_prefix}.ply", 
             points0.detach().cpu().numpy(), 
             colors.cpu().numpy())
    save_ply(f"{validated_point_cloud_path}/{iteration}_{next_image_name_prefix}.ply", 
             points1.detach().cpu().numpy(), 
             colors.cpu().numpy())


def compute_adaptive_threshold(magnitude, min_matches=50, collect_stats=False, beta=0.8):
    """
    Compute adaptive threshold based on IQR.
    
    Args:
        magnitude: torch.Tensor, input magnitude tensor
        min_matches: int, minimum number of valid samples
        collect_stats: bool, whether to collect runtime statistics
        beta: float, EWMA coefficient
        
    Returns:
        float, computed adaptive threshold
    """
    if len(magnitude) <= min_matches:
        return float('inf')
        
    # Compute IQR statistics
    q1, q3 = torch.quantile(magnitude, torch.tensor([0.25, 0.75], device=magnitude.device))
    threshold = q3 + 2.0 * (q3 - q1)
    
    # Update runtime statistics
    if collect_stats:
        if not hasattr(compute_adaptive_threshold, 'running_threshold'):
            compute_adaptive_threshold.running_threshold = threshold.item()
        else:
            compute_adaptive_threshold.running_threshold = (beta * compute_adaptive_threshold.running_threshold + (1 - beta) * threshold.item())
        threshold = compute_adaptive_threshold.running_threshold
        
    return threshold


def analyze_matching_statistics(distances, valid_ratio):
    """
    Analyze statistical characteristics of matching point distances and return an appropriate threshold multiplier
    
    Args:
        distances: tensor, distances between 3D point pairs
        valid_ratio: float, ratio of valid matching points
    
    Returns:
        mean_dist: float, mean distance
        std_dist: float, standard deviation
        threshold_multiplier: float, multiplier used to calculate the threshold
    """
    mean_dist = distances.mean().item()
    std_dist = distances.std().item()
    
    # Select threshold multiplier based on valid matching point ratio
    if valid_ratio < 0.3:
        threshold_multiplier = 2.5
    elif valid_ratio < 0.5:
        threshold_multiplier = 2.0    # 95.45%
    else:
        threshold_multiplier = 1.5
    
    return mean_dist, std_dist, threshold_multiplier


def update_ewma_stats(current_data, beta, func_instance):
    """
    Update statistics using Exponentially Weighted Moving Average (EWMA)
    
    Args:
        current_data: torch.Tensor, current data
        beta: float, EWMA coefficient
        func_instance: function, function instance used to store runtime statistics
    
    Returns:
        tuple: (mean_dist, std_dist)
    """
    if not hasattr(func_instance, 'running_mean'):
        func_instance.running_mean = current_data.mean().item()
        func_instance.running_std = current_data.std().item()
    else:
        func_instance.running_mean = (beta * func_instance.running_mean + (1 - beta) * current_data.mean().item())
        func_instance.running_std = (beta * func_instance.running_std + (1 - beta) * current_data.std().item())
    
    mean_dist = func_instance.running_mean
    std_dist = func_instance.running_std
    
    return mean_dist, std_dist


def get_threshold_multiplier(valid_ratio):
    """
    Determine threshold multiplier based on the ratio of valid matching points
    
    Args:
        valid_ratio: float, ratio of valid matching points
    
    Returns:
        float: threshold multiplier
    """
    if valid_ratio < 0.3:
        return 2.5
    elif valid_ratio < 0.5:
        return 2.0
    return 1.5


def compute_distance_stats(distances, valid_ratio, stats_method="direct", beta=0.8, func_instance=None):
    """
    Compute distance statistics
    
    Args:
        distances: torch.Tensor, distance data
        valid_ratio: float, ratio of valid matching points
        stats_method: str, statistical method ["direct", "ewma", "simple"]
        beta: float, EWMA coefficient
        func_instance: function, function instance used to store runtime statistics
    
    Returns:
        tuple: (mean_dist, std_dist, threshold_multiplier)
    """
    if stats_method == "direct":
        mean_dist, std_dist, threshold_multiplier = analyze_matching_statistics(distances, valid_ratio)
    
    elif stats_method == "ewma":
        mean_dist, std_dist = update_ewma_stats(distances, beta, func_instance)
        threshold_multiplier = get_threshold_multiplier(valid_ratio)
    
    else:  # simple method
        mean_dist = distances.mean().item()
        std_dist = distances.std().item()
        threshold_multiplier = 2.0 if valid_ratio < 0.3 else 1.5
        
    return mean_dist, std_dist, threshold_multiplier


def compute_grid_size(height, width, min_grid_size=15, max_grid_size=60, fixed_size=-1):
    """
    Compute appropriate grid size while maintaining the original image aspect ratio.

    Args:
        height (int): Image height
        width (int): Image width
        min_grid_size (int): Minimum grid size (used in adaptive mode)
        max_grid_size (int): Maximum grid size (used in adaptive mode)
        fixed_size (int): Fixed target grid size. If greater than 0, activates fixed size mode.

    Returns:
        tuple: (grid_h, grid_w, h_cells, w_cells) Grid height and width, and number of grids
    """

    # Fixed Size Mode
    if fixed_size > 0:
        # Ensure the input fixed_size is at least 1 to prevent division by zero.
        safe_fixed_size = max(1, fixed_size)
        
        # Calculate how many target-sized grids can fit in the height and width directions.
        # Use integer division to ensure the number of grids is an integer.
        h_cells = max(1, height // safe_fixed_size)
        w_cells = max(1, width // safe_fixed_size)
        
        # Based on the calculated number of grids, reverse-calculate the actual average height and width of each grid.
        # This ensures that the entire image is covered by grids, even if there are indivisible cases.
        grid_h = height // h_cells
        grid_w = width // w_cells
        
        print(f"[REBUTTAL MODE] Using FIXED grid size. Target: ~{safe_fixed_size}x{safe_fixed_size}. Actual: {grid_h}x{grid_w} pixels.")
        return grid_h, grid_w, h_cells, w_cells

    # Adaptive Mode
    # Calculate image aspect ratio
    aspect_ratio = width / height
    
    # Calculate initial number of grids based on minimum grid size
    h_cells = max(1, height // min_grid_size)
    # Calculate number of grids in width direction based on aspect ratio to maintain approximately square grid cells
    w_cells = max(1, int(h_cells * aspect_ratio))
    
    # Based on the number of grids, reverse-calculate the actual size of each grid
    grid_h = height // h_cells
    grid_w = width // w_cells
    
    # Boundary condition check: Ensure that the "leftover" area due to integer division is not too small,
    # which would lead to unreliable statistics for the boundary grids.
    if height % grid_h < min_grid_size and height % grid_h != 0:
        h_cells = max(1, h_cells - 1)
        grid_h = height // h_cells
    if width % grid_w < min_grid_size and width % grid_w != 0:
        w_cells = max(1, w_cells - 1)
        grid_w = width // w_cells
        
    # Size upper limit check: Ensure that the final grid size is not too large, which would lead to excessive smoothing (high bias).
    if grid_h > max_grid_size:
        h_cells = max(1, height // max_grid_size)
        grid_h = height // h_cells
    if grid_w > max_grid_size:
        w_cells = max(1, width // max_grid_size)
        grid_w = width // w_cells
    
    return grid_h, grid_w, h_cells, w_cells


def normalize_depth(depth):
    """
    Normalize depth map to [0,1] range
    Args:
        depth (torch.Tensor): Input depth map [H,W]
    Returns:
        torch.Tensor: Normalized depth map [H,W]
    """
    valid_mask = depth > 0
    if not valid_mask.any():
        return depth
    
    depth_min = depth[valid_mask].min()
    depth_max = depth[valid_mask].max()
    
    normalized = torch.zeros_like(depth)
    normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8)
    
    return normalized


def map_points_to_grid(points, grid_h, grid_w, h_cells, w_cells):
    """
    Map feature points to grid and count points in each grid (vectorized version)
    Args:
        points (torch.Tensor): Feature point coordinates [N,2]
        grid_h (int): Grid height
        grid_w (int): Grid width
        h_cells (int): Number of grids in height direction
        w_cells (int): Number of grids in width direction
    Returns:
        tuple: (grid_indices, point_counts)
            grid_indices: Grid index for each point [N]
            point_counts: Number of points in each grid [h_cells, w_cells]
    """
    # Calculate grid indices (vectorized operation) 
    grid_y = (points[:, 1] // grid_h).long().clamp(0, h_cells-1)  # [N]
    grid_x = (points[:, 0] // grid_w).long().clamp(0, w_cells-1)  # [N]
    grid_indices = grid_y * w_cells + grid_x                      # [N]

    # Use tensor operations to count points
    point_counts = torch.zeros((h_cells, w_cells), dtype=torch.long, device=points.device)
    point_counts.index_put_(indices=(grid_y, grid_x), 
                           values=torch.ones_like(grid_y), 
                           accumulate=True)

    return grid_indices, point_counts


@torch.amp.autocast(device_type='cuda')
def compute_depth_error_map(render_depth, mono_depth, 
                            valid_points, 
                            iteration=20000, corrected_depth_iter=29900, 
                            error_maps_path=None, 
                            min_grid_size=15, max_grid_size=60,
                            image_name="", fixed_grid_size=-1):
    """Compute depth error map (batch version)"""
    os.makedirs(error_maps_path, exist_ok=True)

    height, width = render_depth.shape
    device = render_depth.device

    # Initialization and preprocessing
    grid_h, grid_w, h_cells, w_cells = compute_grid_size(height, width, min_grid_size, max_grid_size, fixed_grid_size)
    point_grid_indices, point_counts = map_points_to_grid(valid_points, grid_h, grid_w, h_cells, w_cells)

    valid_grid_mask = point_counts >= 5
    valid_grid_indices_flat = torch.nonzero(valid_grid_mask.view(-1), as_tuple=True)[0]
    num_valid_grids = len(valid_grid_indices_flat)

    error_grid = torch.ones((h_cells, w_cells), device=device, dtype=torch.float32)
    scale_grid = torch.ones((h_cells, w_cells), device=device, dtype=torch.float32)
    shift_grid = torch.zeros((h_cells, w_cells), device=device, dtype=torch.float32)
    valid_param_mask = torch.zeros_like(scale_grid, dtype=torch.bool)

    if num_valid_grids == 0:
        return torch.ones_like(render_depth), scale_grid, shift_grid, valid_param_mask

    # Data preparation and padding
    points_in_valid_grids_mask = torch.isin(point_grid_indices, valid_grid_indices_flat)
    
    # Force type conversion outside the autocast(enabled=False) context to ensure type consistency in subsequent operations
    point_coords = valid_points.long()[points_in_valid_grids_mask]
    render_depths_sampled = render_depth[point_coords[:, 1], point_coords[:, 0]]
    mono_depths_sampled = mono_depth[point_coords[:, 1], point_coords[:, 0]]
    
    valid_depth_mask = (render_depths_sampled > 0) & (mono_depths_sampled > 0)
    render_depths = render_depths_sampled[valid_depth_mask]
    mono_depths = mono_depths_sampled[valid_depth_mask]
    filtered_grid_indices = point_grid_indices[points_in_valid_grids_mask][valid_depth_mask]

    # Add robustness check for edge cases
    if filtered_grid_indices.numel() == 0:
        return torch.ones_like(render_depth), scale_grid, shift_grid, valid_param_mask

    filtered_point_counts = torch.bincount(filtered_grid_indices, minlength=h_cells*w_cells).view(h_cells, w_cells)
    
    # Add robustness check for edge cases
    # If all valid grids have zero points after filtering invalid depths, max() will raise an error
    if filtered_point_counts[valid_grid_mask].numel() == 0:
        return torch.ones_like(render_depth), scale_grid, shift_grid, valid_param_mask
    max_points_per_grid = filtered_point_counts[valid_grid_mask].max().item()
    if max_points_per_grid == 0:
        return torch.ones_like(render_depth), scale_grid, shift_grid, valid_param_mask

    batched_mono_depths = torch.zeros(num_valid_grids, max_points_per_grid, device=device)
    batched_render_depths = torch.zeros(num_valid_grids, max_points_per_grid, device=device)
    attention_mask = torch.zeros(num_valid_grids, max_points_per_grid, device=device)

    for i, grid_idx in enumerate(valid_grid_indices_flat):
        mask = (filtered_grid_indices == grid_idx)
        num_pts = mask.sum()
        if num_pts > 0:
            batched_mono_depths[i, :num_pts] = mono_depths[mask]
            batched_render_depths[i, :num_pts] = render_depths[mask]
            attention_mask[i, :num_pts] = 1.0

    # Batch solving (executed in an independent, type-safe environment)
    # Force all type-sensitive linear algebra operations within an autocast(enabled=False) context
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # Force all inputs to float32
        X_mono = batched_mono_depths.float()
        Y_render = batched_render_depths.float()
        mask = attention_mask.float()

        X = torch.stack([X_mono, torch.ones_like(X_mono)], dim=-1)
        Y = Y_render.unsqueeze(-1)

        X = X * mask.unsqueeze(-1)
        Y = Y * mask.unsqueeze(-1)
        
        Xt = X.transpose(1, 2)
        XtX = torch.bmm(Xt, X)
        XtY = torch.bmm(Xt, Y)

        reg = torch.eye(2, device=device, dtype=torch.float32).unsqueeze(0) * 1e-6
        
        # Now XtX and XtY are guaranteed to be float32
        solution = torch.linalg.solve(XtX + reg, XtY)

        scales = solution[:, 0, 0]
        shifts = solution[:, 1, 0]

        pred_depths = X_mono * scales.view(-1, 1) + shifts.view(-1, 1)
        errors = torch.abs(pred_depths - Y_render)
        masked_errors = errors * mask
        mean_errors = masked_errors.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    # Result aggregation and scattering
    grid_y, grid_x = torch.nonzero(valid_grid_mask, as_tuple=True)
    error_grid[grid_y, grid_x] = mean_errors.to(error_grid.dtype)
    
    if iteration >= corrected_depth_iter:
        scale_grid[grid_y, grid_x] = scales.to(scale_grid.dtype)
        shift_grid[grid_y, grid_x] = shifts.to(shift_grid.dtype)
        valid_param_mask[grid_y, grid_x] = True

    error_map = F.interpolate(error_grid.unsqueeze(0).unsqueeze(0),
                            size=(height, width),
                            mode='nearest').squeeze()
    error_map = normalize_non_one_elements(error_map)

    if iteration % 1000 == 0:
        save_error_map(error_map, iteration, error_maps_path, image_name)
    
    return error_map, scale_grid, shift_grid, valid_param_mask


def normalize_non_one_elements(error_map):
    # Create a mask to exclude 1 and invalid values
    valid_mask = (error_map != 1) & torch.isfinite(error_map)
    
    if valid_mask.any():
        valid_values = error_map[valid_mask]
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val > min_val:
            error_map[valid_mask] = (valid_values - min_val) / (max_val - min_val + 1e-8)
    
    return error_map


def save_single_map(data, filename):
    """
    Save a single image.
    Args:
        data: Data to be saved [H,W]
        filename: Path to save the file
    """
    height, width = data.shape
    dpi = 300  # Set a higher DPI
    figsize = (width/dpi, height/dpi)  # Maintain pixel-perfect matching
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    im = ax.imshow(data)
    plt.savefig(filename, bbox_inches=None, pad_inches=0, dpi=dpi)
    plt.close(fig)


def save_error_map(error_map, iteration, error_maps_path, image_name):
    """
    Save error map
    Args:
        error_map (torch.Tensor): Error map [H,W]
        iteration (int): Iteration count
        error_maps_path (str): Path to save
    """    
    os.makedirs(error_maps_path, exist_ok=True)

    # Detach gradients and convert to numpy
    error_np = error_map.detach().cpu().numpy()
    
    # Save raw data
    image_name_prefix = extract_filename_without_extension(image_name)
    np.save(f"{error_maps_path}/{iteration}_{image_name_prefix}_error_map.npy", error_np)
    
    # Save visualization results
    try:
        save_error_map_for_publication(error_np, f"{error_maps_path}/{iteration}_{image_name_prefix}_error_map.png")
    except:
        try:
            save_error_map_with_colorbar(error_np, f"{error_maps_path}/{iteration}_{image_name_prefix}_error_map.png")
        except:
            save_single_map(error_np, f"{error_maps_path}/{iteration}_{image_name_prefix}_error_map.png")


def save_error_map_with_colorbar(data, filename):
    """
    Save error map with a quantitative colorbar legend.
    This function is specifically designed to generate publication-quality, interpretable visualizations.
    """
    # Ensure input is a numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    height, width = data.shape
    dpi = 300
    
    # Create an appropriate figure size to leave space for the colorbar
    figsize = (width / dpi * 1.1, height / dpi) 
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Render the error map using imshow.
    # It is perceptually uniform, with dark colors representing low values and bright colors representing high values.
    # vmin and vmax ensure the color mapping range is fixed between [0, 1], consistent with our normalized errors.
    im = ax.imshow(data, cmap='viridis', vmin=0.0, vmax=1.0)
    
    # Turn off the main plot's axis for a clean look
    ax.set_axis_off()
    
    # Add colorbar legend
    # fraction and pad are empirical parameters to adjust the size and spacing of the colorbar for a harmonious appearance
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Depth Error', rotation=270, labelpad=15, fontsize=10)
    
    # Save the image, bbox_inches='tight' automatically trims excess white space
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def save_error_map_for_publication(data, filename, font_size=10):
    """
    Save error map with a quantitative colorbar legend.
    """
    # Ensure input is a numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    font_family_prefs = ['Times New Roman', 'DejaVu Serif', 'serif']
    found_font = None
    for family in font_family_prefs:
        try:
            # Try to find the font, if successful, use it immediately and break the loop
            findfont(FontProperties(family=family))
            found_font = family
            break
        except Exception:
            continue
    
    if found_font is None:
        found_font = 'serif'

    plt.rcParams['font.family'] = found_font
    plt.rcParams['font.size'] = font_size
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    height, width = data.shape
    dpi = 300

    colorbar_width_inch = 0.15
    figsize = (width / dpi, height / dpi)
    fig_width_with_cbar = figsize[0] + colorbar_width_inch * 1.5
    fig, ax = plt.subplots(figsize=(fig_width_with_cbar, figsize[1]), dpi=dpi)
    main_ax_rect = [0, 0, figsize[0] / fig_width_with_cbar, 1]
    cbar_ax_rect = [figsize[0] / fig_width_with_cbar * 1.05, 0.15, colorbar_width_inch / fig_width_with_cbar, 0.7]
    ax.set_position(main_ax_rect)
    cax = fig.add_axes(cbar_ax_rect)

    im = ax.imshow(data, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_ticks([0.0, 0.5, 1.0])
    # cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    # Modify the label for the 1.0 tick to indicate it's a special value
    cbar.ax.set_yticklabels(['0.0', '0.5', 'N/A']) 
    # Add a title to the colorbar for overall context
    cbar.ax.set_title('Error', loc='left', fontsize=font_size-1)
    
    # Save as vector graphic (PDF) and bitmap (PNG)
    pdf_filename = os.path.splitext(filename)[0] + '.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=dpi)
    
    plt.close(fig)
    matplotlib.rcdefaults()