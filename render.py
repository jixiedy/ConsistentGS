import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import gc
from utils.heatmap_utils import save_heatmap_cv2, blend_heatmap_with_image
from utils.mutil_scale_utils import *
from utils.mono_utils import *


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


def render_set(opt, model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, source_path):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    print(f"Render path: {render_path}, exists: {os.path.exists(render_path)}")
    print(f"GT path: {gts_path}, exists: {os.path.exists(gts_path)}")

    if opt.return_dof:
        renders_dof_path, gts_dof_path, assist_path = make_dof_paths(opt, model_path, iteration=iteration)

    save_frame_interval = 2
    max_frames = 2
    save_dir = os.path.join(model_path, "camera_data", name)
    makedirs(save_dir, exist_ok=True)
    processed_frames = set()
    need_mono_depth = True

    # mon_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda()
    mon_model = torch.hub.load('yvanyin/metric3d', "metric3d_vit_large", pretrain=True).cuda()
    mon_model.eval()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        rendering_dict = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, 
                                return_dof=opt.return_dof, 
                                sensor_width=opt.sensor_width, 
                                f_number=opt.f_number, 
                                focal_length=opt.focal_length, 
                                focus_distance=opt.focus_distance,
                                boundary_type=opt.boundary_type, 
                                use_dynamic_focus_distance=opt.use_dynamic_focus_distance, 
                                kernel_type=opt.kernel_type, 
                                max_blur_kernel_size=opt.max_blur_kernel_size, 
                                use_adaptive=opt.use_adaptive,
                                gaussian_sigma_scale=opt.gaussian_sigma_scale,
                                use_dof_gradient_accum=opt.use_dof_gradient_accum,
                                use_depth_dynamic_focus=opt.use_depth_dynamic_focus,
                                source_path=source_path  
                                )
        
        rendering = rendering_dict["render"]
        
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        print(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        print(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if need_mono_depth and len(processed_frames) < max_frames:
            mono_depth0 = process_mono_depth(view, mon_model)
        else:
            if need_mono_depth:
                need_mono_depth = False
            mono_depth0 = None

        if opt.return_dof:
            save_dofs(opt, renders_dof_path, gts_dof_path, assist_path, rendering, rendering_dict, iteration, idx)
            
            if 'gt_image_dof' in rendering_dict:
                gt_image_dof = rendering_dict['gt_image_dof']
            rendered_image_dof = rendering_dict["rendered_image_dof"]
            current_depth = rendering_dict["depth"]
            
            saved = save_frame_data(view, rendering,
                                    mono_depth0, current_depth,
                                    rendered_image_dof, gt_image_dof,
                                    save_dir, save_frame_interval, 
                                    max_frames, processed_frames, 
                                    iteration)
            if saved and len(processed_frames) >= max_frames:
                cleanup_individual_jsons(save_dir)


def render_sets(opt: OptimizationParams, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(opt, dataset.sh_degree, opt)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        source_path = dataset.source_path

        # bg_color = [1,1,1]
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 

        if not skip_train:
            render_set(opt, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, source_path)

        if not skip_test:
            print("save test render images!")
            render_set(opt, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, source_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", default=False, action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(op.extract(args), model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)