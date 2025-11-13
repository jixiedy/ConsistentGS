from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        # group = parser.add_argument_group(name)
        # for key, value in vars(self).items():
        #     shorthand = False
        #     if key.startswith("_"):
        #         shorthand = True
        #         key = key[1:]
        #     t = type(value)
        #     value = value if not fill_none else None 
        #     if shorthand:
        #         if t == bool:
        #             group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
        #         else:
        #             group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
        #     else:
        #         if t == bool:
        #             group.add_argument("--" + key, default=value, action="store_true")
        #         else:
        #             group.add_argument("--" + key, default=value, type=t)

        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            
            t = type(value)
            value = value if not fill_none else None
            
            if t == bool:
                group.add_argument(f"--{key}", dest=key, action="store_true", default=value)
                group.add_argument(f"--no-{key}", dest=key, action="store_false", default=value)
                parser.set_defaults(**{key: value})
            else:
                if shorthand:
                    group.add_argument(f"--{key}", f"-{key[0:1]}", default=value, type=t)
                else:
                    group.add_argument(f"--{key}", default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"  # "cpu" or "cuda"
        self.eval = True
        
        ### dof parameters ###
        self.R_transpose = False   # SS3DM dataset
        self.use_feature_matching_loss = True
        self.use_mono_depth = True
        self.use_dynamic_weight = False
        self.lambda_feature_matching = 0.05
        self.lambda_depth_consistency = 0.005
        self.sample_type = "nearest"    # "nearest"  "bilinear"
        self.collect_stats = False
        self.use_weight = True
        self.min_depth = 0.1
        self.max_depth = 100.0
        self.motion_threshold = 1.0
        self.min_matches = 50
        self.stats_method = "direct"    # "direct", "ewma", "simple"
        self.min_grid_size = 15
        self.max_grid_size = 60
        self.dof_mode = "no_dof"    # "no_dof" or "both"
        self.train_size = 15
        self.depths_mono = "mon_depth_vit_large"
        ### dof parameters ###
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        # self.use_pcs_render = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"

        self.loss_curriculum_stage_1_iters_scale = 0.5

        ### dof parameters ###
        self.return_dof = True                    
        self.use_dynamic_focus_distance = True     # use dynamic focus distance
        self.use_depth_dynamic_focus = True        # use depth dynamic focus

        self.fixed_grid_size=-1
        
        self.sensor_width = 36.0          # mm
        self.focal_length = 50.0          # mm
        self.f_number = 5.6
        self.focus_distance = 10000.0     # mm

        self.boundary_type = 'One_half'    # 'One_third' or 'One_half' or 'Two_thirds' or 'mean_depth'

        self.kernel_type='gaussian'       # 'gaussian' or 'smooth_step' or 'polygon'
        self.max_blur_kernel_size = 7
        self.gaussian_sigma_scale = 20

        self.lambda_dssim_dof = 0.2

        # Parameters related to Gaussian density control
        self.use_dof_gradient_accum = True
        self.dof_quantile = 0.8    

        # save dof outputs
        self.test_dof = "test_dof"
        self.assist_dof = "assist_dof"
        ### dof parameters ###

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
