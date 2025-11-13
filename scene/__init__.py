import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.camera_utils import loadCam

from utils.dof_utils import build_camera_pairs_mapping
from scene.dataset_readers import SceneInfo
from utils.db_utils import find_colmap_database



class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse_dof")):
            print("Loading custom DOF dataset...")
            # path, images, eval, train_test_exp, mode="no_dof", train_size=15, use_feature_matching_loss=False
            scene_info = sceneLoadTypeCallbacks["Custom_DOF"](args.source_path, args.images, args.eval, args.train_test_exp,
                                                              mode=args.dof_mode, use_feature_matching_loss=args.use_feature_matching_loss, 
                                                              depths_mono=args.depths_mono, use_mono_depth=args.use_mono_depth)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp,
                                                          use_feature_matching_loss=args.use_feature_matching_loss, 
                                                          depths_mono=args.depths_mono, use_mono_depth=args.use_mono_depth)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval,
                                                           args.train_test_exp, args.use_feature_matching_loss, 
                                                           depths_mono=args.depths_mono, use_mono_depth=args.use_mono_depth)
        elif os.path.exists(os.path.join(args.source_path, "scenario.pt")):
            print(f"Auto source_path: {args.source_path}")
            scene_info = sceneLoadTypeCallbacks["Auto"](args.R_transpose, args.source_path, args.images, args.depths, args.eval, args.train_test_exp,
                                                        use_feature_matching_loss=args.use_feature_matching_loss, 
                                                        depths_mono=args.depths_mono, use_mono_depth=args.use_mono_depth)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     #random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 在shuffle前保存图像对应关系
        if args.use_feature_matching_loss:
            self.image_pairs = {}  # 用于存储图像名到下一帧图像名的映射
            if scene_info.train_cameras:
                # 构建图像对应关系
                # self.image_pairs = build_camera_pairs_mapping(scene_info.train_cameras)

                # 检查是否为自定义DOF数据集
                is_custom_dof = os.path.exists(os.path.join(args.source_path, "sparse_dof"))
                db_path = find_colmap_database(args.source_path, is_custom_dof)
                # 针对自定义数据集不检查帧号连续性
                # 当 is_custom_dof = True (你的自定义数据集) 时，ot is_custom_dof = False，所以 check_frame_continuity = False，不会检查帧号连续性
                # 当 is_custom_dof = False (其他数据集) 时，not is_custom_dof = True，所以 check_frame_continuity = True，会检查帧号连续性
                self.image_pairs = build_camera_pairs_mapping(
                    scene_info.train_cameras, 
                    check_frame_continuity=not is_custom_dof, 
                    db_path=db_path
                )
                print(f"Building camera pairs with check_frame_continuity={not is_custom_dof}, db_path={db_path}")

        if shuffle:
            if not args.use_feature_matching_loss:
                random.shuffle(scene_info.train_cameras)   # Multi-res consistent random shuffling
                # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            else:
                # 修改shuffle逻辑，确保相同相机的图像在一组
                camera_groups = {}
                for cam in scene_info.train_cameras:
                    # 根据图像所在目录分组
                    camera_dir = os.path.dirname(cam.image_path)
                    if camera_dir not in camera_groups:
                        camera_groups[camera_dir] = []
                    camera_groups[camera_dir].append(cam)
                
                # 对每组相机内的图像单独进行shuffle
                shuffled_cameras = []
                for group in camera_groups.values():
                    random.shuffle(group)    # 只在组内shuffle
                    shuffled_cameras.extend(group)    # 将shuffle后的组添加到结果列表

                # 创建新的SceneInfo对象
                scene_info = SceneInfo(
                    point_cloud=scene_info.point_cloud,
                    train_cameras=shuffled_cameras,
                    test_cameras=scene_info.test_cameras,
                    nerf_normalization=scene_info.nerf_normalization,
                    ply_path=scene_info.ply_path,
                    is_nerf_synthetic=scene_info.is_nerf_synthetic
                )

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def get_next_camera(self, current_camera, scale=1.0):
        """获取当前相机对应的下一帧相机"""
        if not hasattr(self, 'image_pairs') or not self.image_pairs:
            # print(f"No image pairs found, cannot get next camera.")
            return None
        
        current_image_name = current_camera.image_name
        if current_image_name not in self.image_pairs:
            # print(f"Image {current_image_name} not found in image pairs, cannot get next camera.")
            return None
            
        next_image_name = self.image_pairs[current_image_name]
        for cam in self.train_cameras[scale]:  # 使用原始分辨率
            if cam.image_name == next_image_name:
                return cam
        # print(f"Next camera for {current_image_name} not found in train cameras, cannot get next camera.")
        return None