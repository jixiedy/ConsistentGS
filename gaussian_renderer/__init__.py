
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import gc
from utils.dof_utils import dynamic_focus_distance, adaptive_blur_processing
from torch.cuda.amp import autocast


def compute_coc(point2cam_distance: torch.Tensor, focal_length=50, focus_distance=1000, f_number=4.0, 
                sensor_width=36.0, image_width=1920, hyperfocal_factor=10.0, epsilon=1e-6,
                max_blur_kernel_size=31, min_blur_kernel_size=3
                ):
    """
    Compute Circle of Confusion size for depth of field effects.
    
    Uses standard photography formula when focus distance > hyperfocal_factor * focal_length,
    otherwise uses macro formula.
    
    Args:
        point2cam_distance: Object to camera distance (mm)
        focal_length: Lens focal length (mm)
        focus_distance: Focus plane distance (mm)
        f_number: Aperture f-stop (e.g. f/4.0)
        image_width: Image width (pixels)
        sensor_width: Sensor width (mm, 36mm for full frame)
        epsilon: Small value to prevent division by zero
        hyperfocal_factor: Far/near focus boundary coefficient
        max_blur_kernel_size: Maximum blur kernel size
        min_blur_kernel_size: Minimum blur kernel size
    
    Returns:
        CoC diameter in pixels
    """
    
    focus_distance_eff = focus_distance.item() if isinstance(focus_distance, torch.Tensor) else focus_distance
    if focus_distance_eff >= hyperfocal_factor *  focal_length:
        coc = (focal_length**2) * (point2cam_distance - focus_distance) / (f_number * point2cam_distance * focus_distance)
    else:
        coc = (focal_length**2) * (point2cam_distance - focus_distance) / (f_number * point2cam_distance * (focus_distance - focal_length + epsilon))
    
    coc = coc.abs()
    coc = coc * (image_width / sensor_width)  # Convert from mm to pixels

    # Map to valid kernel size range
    if coc.max() < 1.0:
        max_coc = coc.max()
        min_coc = coc.min()
        target_range = max_blur_kernel_size - min_blur_kernel_size
        current_range = max_coc - min_coc + epsilon
        coc = (coc - min_coc) * (target_range / current_range) + min_blur_kernel_size

    return coc


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, 
           separate_sh=False, override_color=None, use_trained_exp=False,
           return_dof=False, 
           sensor_width=36, 
           f_number=4.0, 
           focal_length=50.0, 
           focus_distance=1000.0, 
           boundary_type="One_half", 
           use_dynamic_focus_distance=False,  
           kernel_type="gaussian", 
           max_blur_kernel_size=31, 
           gaussian_sigma_scale=20,
           use_dof_gradient_accum=False,
           use_depth_dynamic_focus=False,
           source_path="",
           depth_only=False
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    dc = None 

    if not depth_only:
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if separate_sh:
                    dc, shs = pc.get_features_dc, pc.get_features_rest
                else:
                    shs = pc.get_features
        else:
            colors_precomp = override_color
    else:
        if separate_sh:
            # separate_sh=True need dc and shs
            dc = pc.get_features_dc
            shs = torch.zeros_like(pc.get_features_rest)
        else:
            # separate_sh=False need colors_precomp
            colors_precomp = torch.zeros((pc.get_xyz.shape[0], 3), device="cuda")

    # if pipe.use_pcs_render:
    #     print("Using PCS render")
    #     scales = scales / 10
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
            
    out = {
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
    }

    eps = 1e-6
    projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach()
    projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach()
    means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
    means3D_depth_max = means3D_depth.max()
    means3D_depth_min = means3D_depth.min()
    means3D_depth = (means3D_depth - means3D_depth_min) / (means3D_depth_max - means3D_depth_min + eps)
    means3D_depth = means3D_depth.repeat(1,3)
    render_depth, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = means3D_depth,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    render_depth = render_depth.mean(dim=0) 
    out['depth'] = render_depth

    if not depth_only:
        # Apply exposure to rendered image (training only)
        if use_trained_exp:
            exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rendered_image = rendered_image.clamp(0, 1)
        out["render"] = rendered_image

        render_depth = render_depth.clamp(0, 1)
        del raster_settings, rasterizer, screenspace_points, radii, means3D_depth

        #  only compute DOF when required and not test view
        if return_dof:
            # bg_color = torch.tensor([20.0, 20.0, 20.0], device=means3D.device)  

            is_ss3dm = "SS3DM" in source_path
            if viewpoint_camera.invdepthmap is not None:
                if is_ss3dm:
                    # SS3DM dataset: cm to mm
                    depth_input = viewpoint_camera.invdepthmap * float(2**16) * 10.0
                else:
                    # Other datasets with depth maps
                    depth_input = viewpoint_camera.invdepthmap * 1000  # m to mm
            else:
                # Rendered depth processing: apply real_world_scale first, then convert to mm
                depth_input = render_depth * (means3D_depth_max - means3D_depth_min + eps) + means3D_depth_min
                depth_input = depth_input * viewpoint_camera.real_world_scale * 1000
            
            view_dir = means3D - viewpoint_camera.camera_center
            point2cam_distance = torch.norm(view_dir, p=2, dim=1) * viewpoint_camera.real_world_scale * 1000
            
            if use_dynamic_focus_distance:
                if use_depth_dynamic_focus:
                    params = dynamic_focus_distance(depth_input, fov=viewpoint_camera.FoVx, sensor_width=sensor_width, f_number=f_number, 
                                                    boundary_type=boundary_type, source_path=source_path)
                else:
                    params = dynamic_focus_distance(point2cam_distance, fov=viewpoint_camera.FoVx, sensor_width=sensor_width, f_number=f_number, 
                                                boundary_type=boundary_type, source_path=source_path)
                
                focal_length = params['focal_length']
                f_number = params['f_number']
                focus_distance = params['focus_distance']
                del params
            
            if viewpoint_camera.gt_dof_image is not None:
                out.update({'gt_image_dof': viewpoint_camera.gt_dof_image})
            else:
                gt_coc = compute_coc(depth_input, focal_length=focal_length, focus_distance=focus_distance, f_number=f_number, 
                                    sensor_width=sensor_width, image_width=viewpoint_camera.image_width, hyperfocal_factor=10.0, epsilon=eps,
                                    max_blur_kernel_size=max_blur_kernel_size, min_blur_kernel_size=3)
                                
                gt_image = viewpoint_camera.original_image.cuda()
                gt_image_dof = adaptive_blur_processing(gt_image, gt_coc, 
                                                        max_blur_kernel_size=max_blur_kernel_size, 
                                                        gaussian_sigma_scale=gaussian_sigma_scale,
                                                        kernel_type=kernel_type)
                
                out.update({'gt_image_dof': gt_image_dof})
                del gt_image, gt_coc, gt_image_dof
            del render_depth, depth_input
            
            render_coc = compute_coc(point2cam_distance, focal_length=focal_length, focus_distance=focus_distance, f_number=f_number, 
                            sensor_width=sensor_width, image_width=viewpoint_camera.image_width, hyperfocal_factor=10.0, epsilon=eps,
                            max_blur_kernel_size=max_blur_kernel_size, min_blur_kernel_size=3)

            if use_dof_gradient_accum:
                render_coc.requires_grad_(True)
                render_coc.retain_grad()
                out.update({'gaussian_coc': render_coc})
            else:
                out.update({'gaussian_coc': render_coc.detach()})

            render_coc_min = render_coc.min()
            render_coc_max = render_coc.max()
                    
            render_coc = (render_coc - render_coc_min) / (render_coc_max - render_coc_min + eps)
            coc_colors = render_coc.unsqueeze(-1).repeat(1, 3)  # [N, 3]

            depth_raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=0,  # Not using spherical harmonics
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=pipe.debug,
                antialiasing=pipe.antialiasing
            )

            depth_rasterizer = GaussianRasterizer(raster_settings=depth_raster_settings)

            rendered_dof, _, _ = depth_rasterizer(
                            means3D=means3D,
                            means2D=means2D,                
                            shs=None,  # Not using spherical harmonics
                            colors_precomp=coc_colors,
                            opacities=opacity,
                            scales=scales,                  
                            rotations=rotations,            
                            cov3D_precomp=cov3D_precomp
                            )
            
            rendered_dof = rendered_dof.clamp(0, 1)
            out.update({'rendered_dof': rendered_dof.mean(dim=0)})
            
            rendered_dof_InverseNormalized = rendered_dof * (render_coc_max - render_coc_min + eps) + render_coc_min
            rendered_dof_InverseNormalized = rendered_dof_InverseNormalized.mean(dim=0)  # Convert to single channel
                    
            rendered_image_dof = adaptive_blur_processing(rendered_image, 
                                                        rendered_dof_InverseNormalized, 
                                                        max_blur_kernel_size=max_blur_kernel_size,
                                                        gaussian_sigma_scale=gaussian_sigma_scale,
                                                        kernel_type=kernel_type)
            out.update({'rendered_image_dof': rendered_image_dof})
        
            del depth_raster_settings, depth_rasterizer, view_dir, point2cam_distance
            del render_coc, coc_colors, rendered_dof, rendered_image_dof
    
    del means3D, means2D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp
    
    return out
