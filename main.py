import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from lpipsPyTorch import lpips
import json


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from time import perf_counter
import numpy as np
import random
from utils.pose_utils import get_camera_from_tensor
import gc
from utils.dof_utils import make_dof_paths, save_dofs
from metrics import evaluate_dof
from utils.loss_utils import compute_feature_match_with_mono_loss
from utils.mutil_scale_utils import *
from utils.mono_utils import * 


def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    colmap_poses = []
    for i in range(len(index_colmap)):
        ind = index_colmap.index(i+1)
        bb=output_poses[ind]
        bb = bb#.inverse()
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(opt, dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # train_cams_init = scene.getTrainCameras().copy()
    # os.makedirs(scene.model_path + 'pose', exist_ok=True)
    # save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.P, train_cams_init)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training progress", initial=first_iter, total=opt.iterations)

    first_iter += 1
    start = perf_counter()
    # iterations = opt.iterations
    log_interval = 1000

    save_frame_interval = 2  # Save frame interval
    max_frames = 3  # Maximum number of saved frames
    processed_frames = set()  # To track processed frames

    save_dir = os.path.join(scene.model_path, "camera_data")
    os.makedirs(save_dir, exist_ok=True)
    error_maps_path = os.path.join(scene.model_path, "error_maps")
    os.makedirs(error_maps_path, exist_ok=True)
    validated_point_cloud_path = os.path.join(scene.model_path, "validated_point_cloud")
    os.makedirs(validated_point_cloud_path, exist_ok=True)

    # opt.densify_until_iter, 0.5 * (opt.iterations)
    error_map_from_iter = 0.5 * (opt.iterations)        #  Add JSON image save interval
    corrected_depth_iter = 0.995 * (opt.iterations)     #  Start saving corrected depth map iteration
    best_psnr = 0.0

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        next_cam = scene.get_next_camera(viewpoint_cam)  # current_cam, next_cam    
        
        # Check if matches file exists
        if next_cam is not None and dataset.use_feature_matching_loss:
            if not viewpoint_cam.matches_path or not os.path.exists(viewpoint_cam.matches_path):
                next_cam = None
            elif not next_cam.matches_path or not os.path.exists(next_cam.matches_path):
                next_cam = None

        if dataset.data_device == "cpu":
            viewpoint_cam.to("cuda")  # Move to GPU before computation
            if next_cam is not None:
                next_cam.to("cuda")      

        mono_depth0 = None
        if dataset.use_mono_depth and iteration >= error_map_from_iter:
            # Use monocular depth estimation model for non-real depth maps
            if viewpoint_cam.invdepthmap is not None:
                mono_depth0 = viewpoint_cam.invdepthmap.unsqueeze(0).detach().clone()
            elif viewpoint_cam.mono_depth is not None:
                mono_depth0 = viewpoint_cam.mono_depth.unsqueeze(0).detach().clone()
            else:
                mono_depth0 = None
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, 
                        return_dof=opt.return_dof, 
                        sensor_width=opt.sensor_width, 
                        f_number=opt.f_number, 
                        focal_length=opt.focal_length, 
                        focus_distance=opt.focus_distance,
                        boundary_type=opt.boundary_type, 
                        use_dynamic_focus_distance=opt.use_dynamic_focus_distance, 
                        kernel_type=opt.kernel_type, 
                        max_blur_kernel_size=opt.max_blur_kernel_size, 
                        gaussian_sigma_scale=opt.gaussian_sigma_scale,
                        use_dof_gradient_accum=opt.use_dof_gradient_accum,
                        use_depth_dynamic_focus=opt.use_depth_dynamic_focus,
                        source_path=dataset.source_path 
                        )
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        current_depth = render_pkg["depth"]
        
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            # image *= alpha_mask
            image = alpha_mask * image.clone()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = torch.tensor(0., device=gt_image.device) 
        ssim_loss = torch.tensor(0., device=gt_image.device) 
        loss = torch.tensor(0., device=gt_image.device)

        Ll1_dof_with_gt = torch.tensor(0., device=gt_image.device) 
        ssim_loss_dof_with_gt = torch.tensor(0., device=gt_image.device)

        if (not opt.return_dof) or (viewpoint_cam.gt_dof_image is None):
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            ssim_loss = 1.0 - ssim_value
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        if opt.return_dof:
            if 'gaussian_coc' in render_pkg:
                gaussian_coc = render_pkg['gaussian_coc']
            if 'gt_image_dof' in render_pkg:
                gt_image_dof = render_pkg['gt_image_dof']
            rendered_image_dof = render_pkg["rendered_image_dof"]
            
            Ll1_dof_with_gt = l1_loss(rendered_image_dof, gt_image_dof)
            if FUSED_SSIM_AVAILABLE:
                ssim_value_dof_with_gt = fused_ssim(rendered_image_dof.unsqueeze(0), gt_image_dof.unsqueeze(0))
            else:
                ssim_value_dof_with_gt = ssim(rendered_image_dof, gt_image_dof)
            ssim_loss_dof_with_gt = 1.0 - ssim_value_dof_with_gt

            loss_dof = (1.0 - opt.lambda_dssim_dof) * Ll1_dof_with_gt + opt.lambda_dssim_dof * ssim_loss_dof_with_gt
            loss += loss_dof

            if iteration >= error_map_from_iter:
                saved = save_frame_data(viewpoint_cam, image,
                                        mono_depth0, current_depth,
                                        rendered_image_dof, gt_image_dof,
                                        save_dir, save_frame_interval, 
                                        max_frames, processed_frames, 
                                        iteration)
                if saved and len(processed_frames) >= max_frames:
                    cleanup_individual_jsons(save_dir)
                
        # If there is a next frame, render the next frame and compute feature matching loss
        if next_cam is not None and (dataset.use_feature_matching_loss or dataset.use_mono_depth):
            render_pkg_next = render(next_cam, 
                                gaussians, 
                                pipe, 
                                bg, 
                                use_trained_exp=dataset.train_test_exp, 
                                separate_sh=SPARSE_ADAM_AVAILABLE, 
                                return_dof=opt.return_dof, 
                                sensor_width=opt.sensor_width, 
                                f_number=opt.f_number, 
                                focal_length=opt.focal_length, 
                                focus_distance=opt.focus_distance,
                                boundary_type=opt.boundary_type, 
                                use_dynamic_focus_distance=opt.use_dynamic_focus_distance, 
                                kernel_type=opt.kernel_type, 
                                max_blur_kernel_size=opt.max_blur_kernel_size, 
                                gaussian_sigma_scale=opt.gaussian_sigma_scale,
                                use_dof_gradient_accum=opt.use_dof_gradient_accum,
                                use_depth_dynamic_focus=opt.use_depth_dynamic_focus,
                                source_path=dataset.source_path,
                                depth_only=True
                                )

            next_depth = render_pkg_next["depth"]
            del render_pkg_next
            
            # Compute feature matching loss
            feature_matching_loss, depth_consistency_loss = compute_feature_match_with_mono_loss(
                                                                camera=viewpoint_cam, 
                                                                next_cam=next_cam,
                                                                depth0=current_depth, 
                                                                depth1=next_depth, 
                                                                mono_depth0=mono_depth0, 
                                                                sample_type=dataset.sample_type, 
                                                                collect_stats=dataset.collect_stats, 
                                                                iteration=iteration,
                                                                min_depth=dataset.min_depth,
                                                                max_depth=dataset.max_depth,
                                                                motion_threshold=dataset.motion_threshold,
                                                                min_matches=dataset.min_matches,
                                                                use_weight=dataset.use_weight,
                                                                stats_method=dataset.stats_method,
                                                                use_mono_depth=dataset.use_mono_depth,
                                                                min_grid_size=dataset.min_grid_size,
                                                                max_grid_size=dataset.max_grid_size,
                                                                error_map_from_iter=error_map_from_iter,
                                                                corrected_depth_iter=corrected_depth_iter,
                                                                error_maps_path=error_maps_path, 
                                                                validated_point_cloud_path=validated_point_cloud_path,
                                                                fixed_grid_size=opt.fixed_grid_size
                                                                )
            
            if dataset.use_feature_matching_loss:      
                loss += dataset.lambda_feature_matching * feature_matching_loss
            
            if dataset.use_mono_depth:
                loss += dataset.lambda_depth_consistency * depth_consistency_loss
            
            if iteration % log_interval == 0:
                print(f"iteration: {iteration}, Ll1: {Ll1.item()}, {(1.0 - opt.lambda_dssim) * Ll1.item()}")
                print(f"iteration: {iteration}, ssim_loss: {ssim_loss.item()}, {opt.lambda_dssim * ssim_loss.item()}")
                print(f"iteration: {iteration}, Ll1_dof_with_gt: {Ll1_dof_with_gt.item()}, {(1.0 - opt.lambda_dssim_dof) * Ll1_dof_with_gt.item()}")
                print(f"iteration: {iteration}, ssim_loss_dof_with_gt: {ssim_loss_dof_with_gt.item()}, {opt.lambda_dssim_dof * ssim_loss_dof_with_gt.item()}")
                print(f"iteration: {iteration}, feature_matching_loss: {feature_matching_loss.item()}, {dataset.lambda_feature_matching * feature_matching_loss}")
                print(f"iteration: {iteration}, depth_consistency_loss: {depth_consistency_loss.item()}, {dataset.lambda_depth_consistency * depth_consistency_loss}")

            del current_depth, next_depth, mono_depth0
            torch.cuda.empty_cache()

        if opt.return_dof and opt.use_dof_gradient_accum:
            coc_regularization = torch.mean(gaussian_coc)
            loss = loss + 1e-8 * coc_regularization  # Add a small weight

        loss.backward()
        iter_end.record()

        # Save gradients immediately after backward
        coc_grad = None
        if opt.return_dof and opt.use_dof_gradient_accum:
            # coc_grad = None
            if gaussian_coc.grad is not None:
                coc_grad = gaussian_coc.grad.clone()

        del render_pkg
        # torch.cuda.synchronize()
        if iteration % log_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log 

            if iteration % 10 == 0:
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp, *dof_params)
            test_metrics = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, renderArgs, dataset.train_test_exp)
            
            current_psnr = test_metrics.get("PSNR", 0.0)
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                print(f"\n[ITER {iteration}] New best model found with PSNR: {best_psnr:.6f}. Saving checkpoint and best results JSON.")
                
                torch.save((gaussians.capture(), iteration), scene.model_path + "/best_chkpnt.pth")

                method_name = f"ours_{iteration}_best"
                best_results_data = {method_name: best_psnr}
                best_results_json_path = os.path.join(scene.model_path, "best_results.json")
                with open(best_results_json_path, 'w') as f:
                    json.dump(best_results_data, f, indent=4)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, coc_grad)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, coc_grad)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration % log_interval == 0:
                print("Number of Gaussian points:", gaussians._xyz.shape[0])
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                model_path = os.path.join(scene.model_path + "chkpnt")
                os.makedirs(model_path, exist_ok=True)
                full_model_path = os.path.join(model_path, str(iteration) + ".pth")
                # torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save((gaussians.capture(), iteration), full_model_path)

        if dataset.data_device == "cpu":
            viewpoint_cam.to("cpu")
            if next_cam is not None:
                next_cam.to("cpu")

        end = perf_counter()
        train_time = end - start
        print(f"{train_time:.2f} seconds.")
        torch.cuda.empty_cache()
        gc.collect()


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_metrics = {}
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        if opt.return_dof:
            renders_dof_path, gts_dof_path, assist_path = make_dof_paths(opt, args.model_path, iteration=iteration)

        for config in validation_configs:
            render_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "gt")

            print(f"Render path: {render_path}, exists: {os.path.exists(render_path)}")
            print(f"GT path: {gts_path}, exists: {os.path.exists(gts_path)}")

            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                if config['name'] == 'test':
                    psnrs_list = []
                    ssims_list = []
                    lpipss_list = []
                    image_names_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    original_device = viewpoint.data_device
                    viewpoint.to("cuda")
                    
                    render_dict = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_dict["render"], 0.0, 1.0)    
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    gt_image = gt_image[0:3, :, :]
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    viewpoint.to(original_device)

                    if opt.return_dof:
                        save_dofs(opt, renders_dof_path, gts_dof_path, assist_path, image, render_dict, iteration, idx)
                    
                    try:
                        torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name))
                        torchvision.utils.save_image(gt_image, os.path.join(gts_path, viewpoint.image_name))
                    except:
                        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    #########
                    l1_val = l1_loss(image, gt_image).mean().double()
                    psnr_val = psnr(image, gt_image).mean().double()

                    # Ensure ssim and lpips functions are available
                    try:
                        ssim_val = ssim(image, gt_image).mean().double()
                    except NameError:
                        ssim_val = torch.tensor(0.0) 
                        print("Warning: SSIM function not available.")
                    
                    try:
                        lpips_val = lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type='vgg').mean().double()
                    except NameError:
                        lpips_val = torch.tensor(0.0)
                        print("Warning: LPIPS function not available.")

                    l1_test += l1_val
                    psnr_test += psnr_val
                    ssim_test += ssim_val
                    lpips_test += lpips_val
                
                    if config['name'] == 'test':
                        psnrs_list.append(psnr_val.item()) # Store scalar value
                        ssims_list.append(ssim_val.item())
                        lpipss_list.append(lpips_val.item())
                        image_names_list.append(viewpoint.image_name)

                num_cameras = len(config['cameras'])
                psnr_test /= num_cameras 
                l1_test /= num_cameras  
                ssim_test /= num_cameras 
                lpips_test /= num_cameras

                if config['name'] == 'test':
                    test_metrics = {
                        "PSNR": psnr_test.item(),
                        "SSIM": ssim_test.item(),
                        "LPIPS": lpips_test.item()
                    }
                                        
                    method_name = f"ours_{iteration}"
                    print(f"\nSSIM: {ssim_test.item():.7f}")
                    print(f"\nPSNR: {psnr_test.item():.7f}")
                    print(f"\nLPIPS: {lpips_test.item():.7f}")

                    results_json_path = os.path.join(scene.model_path, "results_lossless.json")
                    per_view_json_path = os.path.join(scene.model_path, "per_view_lossless.json")

                    # Load existing data or initialize empty dict
                    try:
                        with open(results_json_path, 'r') as f:
                            full_results_data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        full_results_data = {}

                    try:
                        with open(per_view_json_path, 'r') as f:
                            per_view_results_data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        per_view_results_data = {}

                    # Update data with current iteration's results
                    full_results_data[method_name] = {
                        "SSIM": ssim_test.item(),
                        "PSNR": psnr_test.item(),
                        "LPIPS": lpips_test.item()
                    }
                    per_view_results_data[method_name] = {
                        "SSIM": {name: ssim_val for name, ssim_val in zip(image_names_list, ssims_list)},
                        "PSNR": {name: psnr_val for name, psnr_val in zip(image_names_list, psnrs_list)},
                        "LPIPS": {name: lpips_val for name, lpips_val in zip(image_names_list, lpipss_list)}
                    }

                    # Save updated data back to JSON
                    try:
                        with open(results_json_path, 'w') as f:
                            json.dump(full_results_data, f, indent=4)
                        print(f"Saved/Updated aggregated results to {results_json_path}")
                    except Exception as e:
                        print(f"Error saving results.json: {e}")

                    try:
                        with open(per_view_json_path, 'w') as f:
                            json.dump(per_view_results_data, f, indent=4)
                        print(f"Saved/Updated per-view results to {per_view_json_path}")
                    except Exception as e:
                        print(f"Error saving per_view.json: {e}")
                
                else: # Keep original print format for train set sample evaluation
                    # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    print("\n[ITER {}] Evaluating {}: L1 {:.7f} PSNR {:.7f} SSIM {:.7f} LPIPS {:.7f}".format(
                        iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_metrics


def save_params(args, filename="params.json"):
    params_dict = vars(args)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(params_dict, f, indent=4)
    print(f"Parameters saved to {filename}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    save_params(args, f"{args.model_path}/config_params.json")
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    command = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} python {' '.join(sys.argv)}"
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, 'command.txt'), 'w') as f:
        f.write(command)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    opt = op.extract(args)
    dof_params = [opt.return_dof, 
                  opt.sensor_width, 
                  opt.f_number, 
                  opt.focal_length, 
                  opt.focus_distance, 
                  opt.boundary_type, 
                  opt.use_dynamic_focus_distance,  
                  opt.kernel_type, 
                  opt.max_blur_kernel_size, 
                  opt.gaussian_sigma_scale, 
                  opt.use_dof_gradient_accum,
                  opt.use_depth_dynamic_focus,
                  lp.extract(args).source_path]
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
