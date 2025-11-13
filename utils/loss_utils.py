import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

import os
import numpy as np
import torch.nn as nn
from torchvision import models
from utils.graphics_utils import fov2focal

from utils.feature_matching_utils import sample_depths, save_debug_pointcloud, compute_adaptive_threshold, compute_distance_stats
from utils.feature_matching_utils import process_depth_and_points, process_mono_depth, compute_depth_error_map



C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
    
    
def compute_depth_consistency_loss(error_map, depth_render, depth_mono, lambda_corr=0.2):
    """
    Calculate depth consistency loss
    """

    # Get valid region mask (areas with recovered depth) 
    valid_mask = (error_map > 0) & (depth_render > 0) & (depth_mono > 0)
    
    if not valid_mask.any():
        return (depth_render * 0.0).mean()
    
    absolute_loss = error_map[valid_mask].mean()
    
    # Calculate correlation loss for invalid regions
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        depth_render_invalid = depth_render[invalid_mask]
        depth_render_invalid_max = depth_render_invalid.max()
        depth_render_invalid_min = depth_render_invalid.min()
        
        depth_mono_invalid = depth_mono[invalid_mask]
        depth_mono_invalid_max = depth_mono_invalid.max()
        depth_mono_invalid_min = depth_mono_invalid.min()
        
        depth_render_norm = (depth_render_invalid - depth_render_invalid_min) / (depth_render_invalid_max - depth_render_invalid_min + 1e-8)
        depth_mono_norm = (depth_mono_invalid - depth_mono_invalid_min) / (depth_mono_invalid_max - depth_mono_invalid_min + 1e-8)
        correlation_loss = torch.abs(1 - torch.mean(depth_render_norm * depth_mono_norm))
    else:
        correlation_loss = (depth_render * 0.0).mean()
    
    return absolute_loss + lambda_corr * correlation_loss


def compute_feature_match_with_mono_loss(
                                        camera, 
                                        next_cam, 
                                        depth0=None, 
                                        depth1=None, 
                                        mono_depth0=None, 
                                        sample_type="nearest", 
                                        collect_stats=False,
                                        iteration=20000, 
                                        min_depth=0.1,
                                        max_depth=100.0,
                                        motion_threshold=1.0, 
                                        min_matches=50,
                                        beta=0.8, 
                                        use_weight=False,
                                        stats_method="direct",
                                        use_mono_depth=False,
                                        min_grid_size=15, 
                                        max_grid_size=60,
                                        error_map_from_iter=20000,
                                        corrected_depth_iter=29900,
                                        error_maps_path=None, 
                                        validated_point_cloud_path=None,
                                        fixed_grid_size=-1
                                        ):
    """
    Calculate feature matching loss (supports monocular depth)
    """

    zero_loss = (depth0 * 0.0).mean()
    if not hasattr(camera, 'matching_feature_points') or not camera.matching_feature_points:
        print("Warning: no matches data found")
        return (zero_loss, zero_loss)
    
    # Get matching feature points
    keypoints0 = torch.tensor(camera.matching_feature_points['keypoints0'], device=camera.data_device)
    keypoints1 = torch.tensor(camera.matching_feature_points['keypoints1'], device=camera.data_device)
    
    # Process depth and 3D points
    image0_shape = camera.matching_feature_points['image0_shape']
    image1_shape = camera.matching_feature_points['image1_shape']
    
    # Process rendered depth maps and feature points
    depth0_proc, points3d_world0, keypoints0_scaled, depths0 = process_depth_and_points(depth0, keypoints0, image0_shape, camera, sample_type)
    depth1_proc, points3d_world1, keypoints1_scaled, depths1 = process_depth_and_points(depth1, keypoints1, image1_shape, next_cam, sample_type)
    
    # Process monocular depth map (if enabled)
    if use_mono_depth and iteration >= error_map_from_iter:
        if mono_depth0 is None:
            # Disable depth consistency constraint
            print(f"Warning: Mono depth not available, skipping depth consistency constraint")
            use_mono_depth = False  # Dynamically disable monocular depth constraint
            # Continue calculating basic feature matching loss
        else:
            mono_depth0 = F.interpolate(mono_depth0, size=depth0.shape, mode='bilinear', align_corners=True).squeeze(0)
            mono_depth0 = mono_depth0.squeeze()  # [H,W]
            
            if mono_depth0.shape != depth0.shape:
                raise ValueError(f"Shape mismatch after resize: mono_depth0 {mono_depth0.shape} != depth0 {depth0.shape}")
                
            mono_depth0_proc, mon_depths0 = process_mono_depth(mono_depth0, keypoints0, image0_shape, camera, sample_type)
            
            # Update depth_mask to include monocular depth constraint
            depth_mask = ((depths0 > min_depth) & (depths0 < max_depth) &
                        (depths1 > min_depth) & (depths1 < max_depth) &
                        (mon_depths0 > min_depth) & (mon_depths0 < max_depth))    # [N]
    else:
        depth_mask = ((depths0 > min_depth) & (depths0 < max_depth) &
                    (depths1 > min_depth) & (depths1 < max_depth))    # [N]
        
    boundary_mask = ((keypoints0_scaled[:, 0] >= 0) & 
                    (keypoints0_scaled[:, 0] < depth0_proc.shape[1]) &
                    (keypoints0_scaled[:, 1] >= 0) & 
                    (keypoints0_scaled[:, 1] < depth0_proc.shape[0]) &
                    (keypoints1_scaled[:, 0] >= 0) & 
                    (keypoints1_scaled[:, 0] < depth1_proc.shape[1]) &
                    (keypoints1_scaled[:, 1] >= 0) & 
                    (keypoints1_scaled[:, 1] < depth1_proc.shape[0]))    # [N]
    
    if use_mono_depth and iteration >= error_map_from_iter:
        boundary_mask = boundary_mask & (
            (keypoints0_scaled[:, 0] < mono_depth0_proc.shape[1]) &
            (keypoints0_scaled[:, 1] < mono_depth0_proc.shape[0]))    # [N]
    
    # Filtering dynamic objects, mismatches, and points with large depth estimation errors.
    motion_vectors = points3d_world1 - points3d_world0
    motion_magnitude = torch.norm(motion_vectors, dim=1)
    motion_threshold = compute_adaptive_threshold(motion_magnitude, min_matches=min_matches, collect_stats=collect_stats, beta=beta)
    motion_mask = motion_magnitude < motion_threshold  # [N]
    
    # Combine all validation conditions
    valid_mask = depth_mask & boundary_mask & motion_mask
    
    # Get valid points
    valid_points0 = points3d_world0[valid_mask]
    valid_points1 = points3d_world1[valid_mask]
    valid_keypoints = keypoints0_scaled[valid_mask]
    
    # Check number of valid points
    if len(valid_points0) < min_matches or len(valid_points1) < min_matches:
        print(f"\nWarning: Not enough valid points: {len(valid_points0)} < {min_matches}")
        return (zero_loss, zero_loss)
    
    distances = torch.norm(valid_points0 - valid_points1, dim=1)
    valid_ratio = torch.sum(valid_mask).float() / len(valid_mask)
    
    # Calculate statistics
    mean_dist, std_dist, threshold_multiplier = compute_distance_stats(distances, valid_ratio, stats_method, beta, compute_feature_match_with_mono_loss)
    threshold = mean_dist + threshold_multiplier * std_dist
    valid_distances = distances[distances <= threshold]
    
    if len(valid_distances) < min_matches:
        return (zero_loss, zero_loss)
    
    # Calculate base loss
    if use_weight:
        weights = 1.0 / (valid_distances + 1e-6)
        weights = weights / weights.sum()
        base_loss = (valid_distances * weights).sum()
    else:
        base_loss = valid_distances.mean()
                    
    # If monocular depth constraint is enabled, calculate additional depth loss
    if use_mono_depth and iteration >= error_map_from_iter:
        # Calculate depth error map
        error_map, scale_grid, shift_grid, valid_param_mask = compute_depth_error_map(
                                            depth0_proc, mono_depth0_proc, 
                                            valid_keypoints, 
                                            iteration, corrected_depth_iter, 
                                            error_maps_path, 
                                            min_grid_size, max_grid_size,
                                            camera.image_name, fixed_grid_size)
        del scale_grid, shift_grid, valid_param_mask
        
        # Calculate depth consistency loss
        depth_consistency_loss = compute_depth_consistency_loss(error_map, depth0_proc, mono_depth0_proc)
        
        if iteration % 1000 == 0 and iteration >= error_map_from_iter:
            save_debug_pointcloud(validated_point_cloud_path, iteration, camera.image_name, next_cam.image_name, valid_points0, valid_points1)
        
        return base_loss, depth_consistency_loss
        
    else:
        if iteration % 1000 == 0 and iteration >= error_map_from_iter:
            save_debug_pointcloud(validated_point_cloud_path, iteration, camera.image_name, next_cam.image_name, valid_points0, valid_points1)
        
        return base_loss, zero_loss