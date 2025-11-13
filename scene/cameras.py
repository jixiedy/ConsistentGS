import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import PILtoTorch
import cv2
import os
from PIL import Image
import gc


class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_path=None,
                 matches_path=None, 
                 matching_feature_points=None, 
                 mono_depth_path=None,
                 c2w_nerf=None,
                 gt_dof_path=None, 
                 real_world_scale = 1.0,
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        self.image_path = image_path
        self.matches_path = matches_path
        self.matching_feature_points = matching_feature_points
        self.c2w_nerf = c2w_nerf

        # Add new attributes for DOF support
        self.gt_dof_image = None
        self.real_world_scale = real_world_scale
        self.is_test_view = is_test_view

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.mono_depth = None
        self.mono_depth_path = mono_depth_path
        if self.mono_depth_path and os.path.exists(self.mono_depth_path) and invdepthmap is None:
            mono_depth = cv2.imread(self.mono_depth_path, -1).astype(np.float32)
            if mono_depth is not None:
                mono_depth = cv2.resize(mono_depth, resolution)
                mono_depth = mono_depth / float(2**16)
                self.mono_depth = torch.from_numpy(mono_depth[None]).to(self.data_device)
                self.mono_depth = self.mono_depth.detach().clone()

        if gt_dof_path and os.path.exists(gt_dof_path):
            dof_image = Image.open(gt_dof_path)
            resized_dof_image = PILtoTorch(dof_image, resolution)
            self.gt_dof_image = resized_dof_image[:3, ...].clamp(0.0, 1.0).to(self.data_device)

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to(self, device):
        self.data_device = torch.device(device)
        if self.alpha_mask is not None:
            self.alpha_mask = self.alpha_mask.to(device)
        if self.original_image is not None:
            self.original_image = self.original_image.to(device)
        if self.invdepthmap is not None:
            self.invdepthmap = self.invdepthmap.to(device)
        if self.gt_dof_image is not None:
            self.gt_dof_image = self.gt_dof_image.to(device)
        if self.mono_depth is not None:
            self.mono_depth = self.mono_depth.to(device)
        return self


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

