import os
from os import makedirs
import math
import numpy as np
from functools import lru_cache
from typing import Dict, Union, List, Tuple

import torch
import torchvision
import torch.nn.functional as F

import re
from pathlib import Path
from scipy.stats import gaussian_kde

from utils.heatmap_utils import save_heatmap_cv2, blend_heatmap_with_image
from utils.graphics_utils import fov2focal 

from utils.db_utils import extract_frame_number, read_colmap_adjacency


def build_camera_pairs_mapping(cam_infos, check_frame_continuity=True, db_path=None):
    """
    Build a mapping of camera pairs using COLMAP adjacency information.
    
    Args:
        cam_infos: List of camera information objects.
        check_frame_continuity: Whether to check for frame number continuity (ignored if db_path is provided).
        db_path: Path to the COLMAP database file.
        
    Returns:
        A dictionary mapping each image name to its next image name.
    """
    if not cam_infos:
        print("Warning: Empty camera info list")
        return {}
    
    if db_path and os.path.exists(db_path):
        print(f"Using COLMAP database at {db_path} for camera pairs")
        # Use COLMAP adjacency information
        adjacency = read_colmap_adjacency(db_path)
        
        # Create a mapping from image name to camera info
        cam_by_name = {cam.image_name: cam for cam in cam_infos}
        
        # Build image pairs
        image_pairs = {}
        for cam in cam_infos:
            img_name = cam.image_name
            
            if img_name not in adjacency:
                continue
            
            # Get the non-test image with the highest match count
            best_match = None
            best_match_count = 0
            
            for adj_img_name, match_count in adjacency[img_name]:
                if adj_img_name in cam_by_name and not cam_by_name[adj_img_name].is_test:
                    if match_count > best_match_count:
                        best_match = adj_img_name
                        best_match_count = match_count
            
            if best_match:
                image_pairs[img_name] = best_match
        
        return image_pairs
    else:
        print("No valid COLMAP database found, using frame continuity for camera pairs")
        # Fallback to the original method
        camera_groups = {}
        
        # Group by directory
        for cam in cam_infos:
            camera_dir = os.path.dirname(cam.image_name)
            if camera_dir not in camera_groups:
                camera_groups[camera_dir] = []
            camera_groups[camera_dir].append(cam)
        
        image_pairs = {}
        for dir_path, cameras in camera_groups.items():
            # Extract frame number and sort
            frame_info = []
            for cam in cameras:
                frame_num = extract_frame_number(cam.image_name)
                if frame_num != -1:
                    frame_info.append((frame_num, cam))
            
            # Sort by frame number
            frame_info.sort(key=lambda x: x[0])
            
            # Build mapping between consecutive frames
            for i in range(len(frame_info) - 1):
                curr_frame, curr_cam = frame_info[i]
                next_frame, next_cam = frame_info[i + 1]
                
                # Check frame continuity if required
                if not check_frame_continuity or next_frame - curr_frame == 1:
                    # Don't include pairs where the next camera is in the test set
                    if not next_cam.is_test:
                        image_pairs[curr_cam.image_name] = next_cam.image_name
        
        return image_pairs


def mono_channelization_map(input_map: torch.Tensor) -> torch.Tensor:
    if input_map.ndim not in [2, 3]:
        raise ValueError(f"Unexpected depth map dimensions: {input_map.ndim}")
        
    if input_map.ndim == 3:
        if input_map.shape[0] == 1:
            input_map = input_map.squeeze(0)
        elif input_map.shape[-1] == 1:
            input_map = input_map.squeeze(-1)
        elif input_map.shape[0] in [3, 4]:
            input_map = input_map.mean(dim=0)
        elif input_map.shape[-1] in [3, 4]:
            input_map = input_map.mean(dim=-1)
        else:
            raise ValueError(f"Invalid depth map shape: {input_map.shape}")
            
    return input_map.float()


@lru_cache(maxsize=10)
def get_cached_gaussian_kernel(k, sigma=None, gaussian_sigma_scale=10, device="cuda"):
    return create_gaussian_blur_kernel(k, sigma=sigma, gaussian_sigma_scale=gaussian_sigma_scale, device=device)


def create_smooth_step_blur_kernel(diameter, device="cuda"):
    """
    Create a Smooth Step Blur Kernel
    Args:
        diameter: Diameter of the blur kernel
        device: Computational device
    Returns:
        kernel: Blur kernel matrix [diameter, diameter]
    """
    # Ensure diameter is odd
    diameter = int(round(diameter))
    diameter = diameter if diameter % 2 == 1 else diameter + 1

    # Create coordinate grid
    radius = diameter // 2
    x = torch.arange(-radius, radius + 1, device=device)
    y = torch.arange(-radius, radius + 1, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Calculate squared radius
    R2 = X**2 + Y**2
    r_squared = radius**2

    # Compute blur kernel K_s(x, y, r)
    kernel = 0.5 + 0.5 * torch.tanh(0.25 * (r_squared - R2) + 0.5)

    # Normalize
    kernel = kernel.clamp(min=0)
    kernel /= kernel.sum()
    return kernel


def create_gaussian_blur_kernel(diameter, sigma=None, gaussian_sigma_scale=20, device="cuda"):
    """
    Create a 2D Gaussian blur kernel
    Args:
        diameter: Diameter of the blur kernel (kernel size)
        sigma: Standard deviation of the Gaussian distribution
               If None, will be calculated based on kernel size
        device: Computational device
    Returns:
        kernel: Gaussian blur kernel matrix [diameter, diameter]
    """
    # Ensure diameter is odd
    diameter = int(round(diameter))
    diameter = diameter if diameter % 2 == 1 else diameter + 1
    
    # Calculate sigma if not provided
    if sigma is None:
        sigma = (diameter - 1) / gaussian_sigma_scale
    
    # Create coordinate grid
    radius = diameter // 2
    x = torch.arange(-radius, radius + 1, device=device)
    y = torch.arange(-radius, radius + 1, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate 2D Gaussian
    Z = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Normalize
    kernel = Z / Z.sum()
    return kernel


def create_polygon_blur_kernel(diameter, num_aperture_blades=8, device="cuda"):
    """
    Create a polygonal blur kernel.
    """
    
    # Ensure diameter is odd
    diameter = int(round(diameter))
    diameter = diameter if diameter % 2 == 1 else diameter + 1
    radius = diameter // 2
    
    # Generate vertices of a regular polygon
    angles = torch.linspace(0, 2*torch.pi, num_aperture_blades, device=device)
    vertices = torch.stack([
        radius * torch.cos(angles),
        radius * torch.sin(angles)
    ], dim=1)
    
    # Create grid points
    x = torch.linspace(-radius, radius, diameter, device=device)
    y = torch.linspace(-radius, radius, diameter, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Use cross product to determine if points are inside the polygon
    inside = torch.zeros(points.shape[0], dtype=torch.float32, device=device)
    
    # Close the polygon
    vertices = torch.cat([vertices, vertices[0:1]], dim=0)
    # Check each edge
    for i in range(num_aperture_blades):
        v1, v2 = vertices[i], vertices[i+1]
        
        # Calculate edge vector and vector from first vertex to points
        edge = v2 - v1
        to_point = points - v1.unsqueeze(0)
        
        # Calculate cross product
        cross_product = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]
        
        # Update inside flag
        inside += (cross_product < 0).float()
    
    # If the inside count equals the number of edges, the point is inside the polygon
    kernel = (inside == num_aperture_blades).float().reshape(diameter, diameter)
    
    # Add radial attenuation
    R = torch.sqrt(X**2 + Y**2)
    radial_weight = torch.cos(0.5 * torch.pi * R / radius).clamp(min=0)
    # radial_weight = (1 - (R / radius).clamp(0, 1)).pow(2)
    kernel = kernel * radial_weight
    
    # Ensure kernel has enough non-zero values before normalization
    if kernel.sum() > 1e-6:  # Add numerical stability check
        kernel = kernel / kernel.sum()
    else:
        # If the sum is too small, use an alternative
        kernel = torch.ones_like(kernel) / kernel.numel()
    
    return kernel


def dynamic_padding(kernel_size):
    """Calculate dynamic padding"""
    return torch.div(kernel_size - 1, 2, rounding_mode='floor')


def ensure_odd_size(tensor):
    """Ensure kernel size is odd"""
    return torch.div(tensor, 2, rounding_mode='floor') * 2 + 1


# ====================================================================================
# REBUTTAL REFACTORING MODIFICATION: START
# Scientific basis: This function is a complete refactoring of "Bottleneck 2: Naive Defocus Convolution". 
# It replaces the original inefficient implementation based on Python loops and multiple full-image convolutions 
# with a high-performance paradigm of "im2col + batch matrix multiplication". 
# This method is mathematically equivalent to the original implementation but improves computational efficiency by an order of magnitude.
# ====================================================================================
def adaptive_blur_processing(rgb_image, kernel_benchmark_map, max_blur_kernel_size=31, 
                           min_blur_kernel_size=3, gaussian_sigma_scale=20, kernel_type="gaussian"):
    """
    High-performance adaptive blur processing function based on im2col.
    This function efficiently simulates physically accurate defocus effects
    by transforming the spatially varying convolution problem into a single large-scale batch matrix operation.
    """
    # Initialization and preprocessing
    rgb_image = rgb_image.contiguous().float()
    
    # Ensure input tensor continuity and type consistency
    device = rgb_image.device
    dtype = rgb_image.dtype
    num_channels, H, W = rgb_image.shape
    
    # Calculate the required blur kernel size for each pixel and ensure it is odd
    kernel_benchmark_map = mono_channelization_map(kernel_benchmark_map).to(device).contiguous().float()
    kernel_sizes = torch.round(kernel_benchmark_map).clamp(min=min_blur_kernel_size, max=max_blur_kernel_size)
    kernel_sizes = ensure_odd_size(kernel_sizes).long()

    # Determine maximum kernel size and padding
    # Find the maximum value among all required blur kernel sizes; subsequent operations are based on this maximum size
    unique_sizes = torch.unique(kernel_sizes)
    if len(unique_sizes) == 0: # Edge case handling 
        return rgb_image
    max_k = unique_sizes.max().item()
    max_padding = (max_k - 1) // 2

    # Extract all image patches (im2col)
    # Pad the input image once, then use F.unfold to extract the neighborhood patches for each pixel
    padded_input = F.pad(rgb_image.unsqueeze(0), pad=(max_padding, max_padding, max_padding, max_padding), mode='reflect')
    unfolded_patches = F.unfold(padded_input, kernel_size=(max_k, max_k))
    # Reshape to match each pixel: (N, C, k*k), where N = H*W
    unfolded_patches = unfolded_patches.view(num_channels, max_k * max_k, -1).permute(2, 0, 1)

    # Construct kernel lookup table (Kernel Bank)
    # Create a "kernel bank" that generates all required blur kernels of different sizes at once
    kernel_bank = torch.zeros(len(unique_sizes), max_k * max_k, device=device, dtype=dtype)

    for i, size_tensor in enumerate(unique_sizes):
        k = size_tensor.item()
        if kernel_type == "gaussian":
            kernel = get_cached_gaussian_kernel(k, sigma=None, gaussian_sigma_scale=gaussian_sigma_scale, device=device).to(dtype=dtype)
        elif kernel_type == "smooth_step":
            kernel = create_smooth_step_blur_kernel(k, device=device).to(dtype=dtype)
        elif kernel_type == "polygon":
            kernel = create_polygon_blur_kernel(k, num_aperture_blades=8, device=device).to(dtype=dtype)
        
        # Pad the smaller kernel to the maximum size, then flatten and store in the "kernel bank"
        pad_amount = (max_k - k) // 2
        padded_kernel = F.pad(kernel, (pad_amount, pad_amount, pad_amount, pad_amount))
        kernel_bank[i] = padded_kernel.flatten()

    # Batch indexing and lookup
    # Create a mapping tensor from kernel sizes to "kernel bank" indices for fast lookup
    max_size_val = kernel_sizes.max().item()
    mapping_tensor = torch.full((max_size_val + 1,), -1, dtype=torch.long, device=device)
    unique_sizes_cpu = unique_sizes.cpu().numpy()
    indices_cpu = np.arange(len(unique_sizes_cpu))
    mapping_tensor[unique_sizes_cpu] = torch.from_numpy(indices_cpu).to(device)
    
    # Find the corresponding "kernel bank" index for each pixel
    kernel_indices = mapping_tensor[kernel_sizes.flatten()]
    
    # Use F.embedding (an efficient gather operation) to fetch the corresponding blur kernel for each pixel in parallel
    pixel_kernels = F.embedding(kernel_indices, kernel_bank)

    # Batch convolution (implemented via einsum)
    # This is the core of the entire optimization. Using Einstein summation convention,
    # batch dot product N image patches of shape (C, k*k)
    # with N kernels of shape (k*k), resulting in N vectors of shape (C,).
    # 'nci,ni->nc' means: for each pixel n, multiply its C channel image patches i with the kernel i and sum.
    result_flat = torch.einsum('nci,ni->nc', unfolded_patches, pixel_kernels)

    # Reshape the result
    # Reshape the computed (N, C) result vectors back to the image format (C, H, W)
    processed_image = result_flat.permute(1, 0).view(num_channels, H, W)
    
    return processed_image.clamp(0, 1)


class FocusParameters:
    """
    Camera sensor specifications, aperture, and focus distance preset parameters
    """
    # Common camera sensor size options (mm)
    CAMERA_SENSOR = [3.6, 4.8, 5.27, 6.4, 7.2, 10.8, 13.2, 17.3, 23.5, 27.9, 36, 44, 53.7]
    # Common aperture values
    F_NUMBERS = [1.0, 1.2, 1.4, 1.8, 2.0, 2.8, 4.0, 5.6, 8.0, 10.0, 11.0, 16.0]
    
    # Minimum focus distance (mm)
    MIN_FOCUS_DISTANCE = 100  # waymo: 500; mip-nerf 360: 100; 

    DEFAULT_PARAMS = {
        'focal_length': 50.0,
        'f_number': 4.0,
        'focus_distance': 10000.0  # 10 meters
    }


def calculate_depth_distribution(depth_map: torch.Tensor, source_path: str = "") -> Dict[str, float]:
    """
    Calculate the distribution characteristics of the depth map and choose different filtering strategies based on the dataset type
    """
    # Flatten the depth values
    depth_flat = depth_map.flatten()
    
    # Choose filtering strategy based on dataset type
    is_ss3dm = "SS3DM" in source_path
    
    if is_ss3dm:
        # SS3DM dataset uses a fixed threshold to filter the background
        depth_flat = depth_flat[depth_flat < 655340]
    else:
        # Other datasets use a more appropriate threshold in millimeters
        # 65536000 = 2^16(max depth value) * 1000(meters to millimeters)
        depth_flat = depth_flat[(depth_flat > 0) & (depth_flat < 65536000)]
    
    # Safety check
    if len(depth_flat) == 0:
        return {
            'mean_depth': 10000.0,  # 10 meters (millimeters)
            'Two_thirds': 10000.0,
            'One_half': 10000.0, 
            'One_third': 10000.0,
        }
    
    depth_sorted = torch.sort(depth_flat)[0]
    total_pixels = len(depth_flat)
    
    stats = {
        'mean_depth': depth_flat.mean().item(),
        'min_depth': depth_sorted[0].item(),
        'max_depth': depth_sorted[-1].item(),
        'Two_thirds': depth_sorted[int(total_pixels*2/3)].item(),
        'One_half': depth_sorted[int(total_pixels/2)].item(),
        'One_third': depth_sorted[int(total_pixels/3)].item(),
    }
    
    return stats


def is_close_to_any(value: float, valid_values: list, rtol: float = 1e-05) -> bool:
    """
    Check if a floating-point number is approximately equal to any value in a list

    Args:
        value: The value to check
        valid_values: List of valid values
        rtol: Relative tolerance
        
    Returns:
        bool: Whether it matches any valid value
    """
    return any(math.isclose(value, valid, rel_tol=rtol) for valid in valid_values)


def dynamic_focus_distance(
    depth_map: torch.Tensor,
    fov: float,
    sensor_width: float,
    f_number: float = 4.0,
    boundary_type: str = 'One_third',
    source_path: str = ""
    ) -> Dict[str, float]:
    """
    Get optimal depth of field parameters
    
    Args:
        depth_map: torch.Tensor, depth map or object distance map
        fov: float, field of view (radians)
        sensor_width: float, sensor size (mm)
        f_number: float, aperture value
        boundary_type: str, boundary type
        scene_preset: str, scene preset name
    """
    # Safety check
    # assert is_close_to_any(float(sensor_width), FocusParameters.CAMERA_SENSOR), f"Invalid sensor width: {sensor_width}"
    # assert is_close_to_any(float(f_number), FocusParameters.F_NUMBERS), f"Invalid f-number: {f_number}"
    if depth_map is None or depth_map.numel() == 0:
        return {
            'focal_length': 50.0,      # mm
            'f_number': f_number,
            'focus_distance': 10000.0  # 10 meters (millimeters)
        }

    focal_length = fov2focal(fov, sensor_width)
    depth_stats = calculate_depth_distribution(depth_map, source_path)
    focus_distance = max(FocusParameters.MIN_FOCUS_DISTANCE, depth_stats[boundary_type])
    
    return {
        'focal_length': focal_length,
        'f_number': f_number,
        'focus_distance': focus_distance,
        # 'depth_stats': depth_stats
        }


def make_dof_paths(opt, model_path, iteration=None):
    renders_dof_path = os.path.join(model_path, opt.test_dof, "dofs_{}".format(iteration), "renders")
    gts_dof_path = os.path.join(model_path, opt.test_dof, "dofs_{}".format(iteration), "gt")
    assist_path = os.path.join(model_path, opt.assist_dof, "assist_dof_{}".format(iteration), "assist")
    
    makedirs(renders_dof_path, exist_ok=True)
    makedirs(gts_dof_path, exist_ok=True)
    makedirs(assist_path, exist_ok=True)

    return renders_dof_path, gts_dof_path, assist_path


def save_dofs(opt, renders_dof_path, gts_dof_path, assist_path, rendering, rendering_dict, iteration, idx):            
    if "rendered_dof" in rendering_dict:  
        rendered_dof = rendering_dict["rendered_dof"]
        
        if rendered_dof.max() > 1.0:
            rendered_dof = (rendered_dof - rendered_dof.min()) / (rendered_dof.max() - rendered_dof.min() + opt.eps)
        else:
            rendered_dof = torch.clamp(rendered_dof, 0.0, 1.0)
        
        torchvision.utils.save_image(rendered_dof, f"{assist_path}/dof_viz_{iteration}_{idx:05d}.png")
        save_heatmap_cv2(rendered_dof, f"{assist_path}/heatmap_dof_viz__{iteration}_{idx:05d}.png")
        blend_heatmap_with_image(rendered_dof, rendering, f"{assist_path}/mix_heatmap_{iteration}_{idx:05d}.png")
    
    if "rendered_image_dof" in rendering_dict:
        rendered_image_dof = rendering_dict["rendered_image_dof"]
        
        if rendered_image_dof.max() > 1.0:
            rendered_image_dof = (rendered_image_dof - rendered_image_dof.min()) / (rendered_image_dof.max() - rendered_image_dof.min() + opt.eps)
        else:
            rendered_image_dof = torch.clamp(rendered_image_dof, 0.0, 1.0)

        torchvision.utils.save_image(rendered_image_dof, f"{renders_dof_path}/rendered_image_dof_{iteration}_{idx:05d}.png")

    if "gt_image_dof" in rendering_dict:
        gt_image_dof = rendering_dict["gt_image_dof"]
        
        if gt_image_dof.max() > 1.0:
            gt_image_dof = (gt_image_dof - gt_image_dof.min()) / (gt_image_dof.max() - gt_image_dof.min() + opt.eps)
        else:
            gt_image_dof = torch.clamp(gt_image_dof, 0.0, 1.0)
        
        torchvision.utils.save_image(gt_image_dof, f"{gts_dof_path}/gt_image_dof_{iteration}_{idx:05d}.png")

    

