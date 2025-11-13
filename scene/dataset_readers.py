import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


import cv2
import pickle
import re
from utils.db_utils import find_colmap_database, read_colmap_adjacency


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    matches_path: str  
    matching_feature_points: np.array  
    mono_depth_path: str  
    c2w_nerf: np.array
    gt_dof_path: str
    real_world_scale: float
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def colmap_to_nerf_transform(R, T, R_transpose=False):
    """Convert COLMAP camera R,T to NeRF-style transformation matrix"""
    # Construct w2c matrix from R,T
    w2c = np.eye(4)
    w2c[:3, :3] = R.T if R_transpose else R # R from COLMAP is already transposed
    w2c[:3, 3] = T
    
    # Invert to get c2w
    c2w = np.linalg.inv(w2c)
    
    # Convert from COLMAP convention back to OpenGL/Blender convention
    c2w[:3, 1:3] *= -1  # Invert Y and Z axes
    
    return c2w


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list, 
                      use_feature_matching_loss=False, depths_mono_folder=""):
    sorted_keys = sorted(cam_extrinsics.keys(), key=lambda k: cam_extrinsics[k].name) if use_feature_matching_loss else cam_extrinsics
    cam_infos = []
    for idx, key in enumerate(sorted_keys):    # cam_extrinsics
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        c2w_nerf = colmap_to_nerf_transform(R, T)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        matches_path = ""
        matching_feature_points = None
        if use_feature_matching_loss and idx < len(sorted_keys) - 1:
            # use the sorted key list to get the next camera info
            next_key = sorted_keys[idx + 1]
            next_extr = cam_extrinsics[next_key]
            
            current_name = extr.name[:-4]
            next_name = next_extr.name[:-4]
            
            # Construct matches file path
            matches_base = os.path.join(os.path.dirname(images_folder), "matches")
            matches_subdir = os.path.dirname(os.path.relpath(image_path, images_folder))
            matches_name = f"{current_name}_{next_name}.npy"
            matches_path = os.path.join(matches_base, matches_subdir, matches_name)
            # print(f"current matches_path: {matches_path}")
            matching_feature_points = np.load(matches_path, allow_pickle=True).item()

        mono_depth_path = os.path.join(depths_mono_folder, f"{extr.name[:-n_remove]}.png") if depths_mono_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              matches_path=matches_path, 
                              matching_feature_points=matching_feature_points, 
                              mono_depth_path=mono_depth_path,
                              c2w_nerf=c2w_nerf,
                              gt_dof_path="", 
                              real_world_scale=1.0,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8, 
                        use_feature_matching_loss=False, depths_mono="", use_mono_depth=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list, 
        use_feature_matching_loss=use_feature_matching_loss,
        depths_mono_folder=os.path.join(path, depths_mono) if depths_mono != "" and use_mono_depth else ""
        )
    
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if use_feature_matching_loss:
        cam_infos = cam_infos_unsorted  # already sorted by name
    else:
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png", use_feature_matching_loss=False, depths_mono_folder=""):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
            mono_depth_path = os.path.join(depths_mono_folder, f"{image_name}.png") if depths_mono_folder != "" else ""

            # Construct matches file path
            matches_path = ""
            matching_feature_points = None
            if use_feature_matching_loss:
                if idx < len(frames) - 1:
                    # Get the current and next frame file names
                    curr_stem = Path(frame["file_path"]).stem 
                    next_stem = Path(frames[idx + 1]["file_path"]).stem
                    
                    # NeRF dataset matches should be placed in the matches subdirectory
                    # For example: path/matches/r_0_r_1.npy
                    matches_name = f"{curr_stem}_{next_stem}.npy"
                    matches_path = os.path.join(path, "matches", matches_name)
                    matching_feature_points = np.load(matches_path, allow_pickle=True).item()

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], 
                            matches_path=matches_path, 
                            matching_feature_points=matching_feature_points, 
                            mono_depth_path=mono_depth_path,
                            c2w_nerf=c2w,
                            gt_dof_path="", 
                            real_world_scale=1.0,
                            depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png", use_feature_matching_loss=False, depths_mono="", use_mono_depth=False):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    depths_mono_folder = os.path.join(path, depths_mono) if depths_mono != "" and use_mono_depth else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension,
                                                use_feature_matching_loss=use_feature_matching_loss,
                                                depths_mono_folder=depths_mono_folder
                                                )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension,
                                               use_feature_matching_loss=use_feature_matching_loss,
                                               depths_mono_folder=depths_mono_folder
                                               )
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


def readAutoSceneInfo(R_transpose: bool, path, images, depths, eval, train_test_exp, llffhold=8, use_feature_matching_loss=False, depths_mono="", use_mono_depth=False):
    with open(os.path.join(path, "scenario.pt"), 'rb') as f:
        scenario_dict = pickle.load(f)

    # Get basic scene information
    n_frames = scenario_dict['metas']['num_frames']
    camera_names = ['BACK_RIGHT', 'BACK_LEFT', 'BACK', 'FRONT_RIGHT', 'FRONT_LEFT', 'FRONT']
    
    # Read all camera information
    cam_infos = []
    for cam_id in range(6):
        camera_name = camera_names[cam_id]
        cam_data = scenario_dict['observers'][f'camera_{camera_name}']['data']

        for frame_id in range(n_frames):
            # Get pose
            c2w = cam_data['c2w'][frame_id]  # camera to world transform
            w2c = np.linalg.inv(c2w)
            T = w2c[:3, 3]
                        
            # Extract Town number from path
            path_parts = path.split(os.sep)
            town_name = None
            for part in path_parts:
                if part.lower().startswith("town"):
                    town_name = part
                    break

            # For all other Towns, based on experiments, transposing R works better
            if cam_id == 0 and frame_id == 0:
                print(f"[INFO] Applying transpose rule for {town_name} based on experimental findings.")
            R = np.transpose(w2c[:3, :3])
            # Correspondingly, pass the correct flag to colmap_to_nerf_transform
            c2w_nerf = colmap_to_nerf_transform(R, T, True)
            
            # get intrinsics
            intr = cam_data['intr'][frame_id]  
            hw = cam_data['hw'][frame_id]
            width, height = int(hw[1]), int(hw[0])
            fx, fy = float(intr[0,0]), float(intr[1,1])
            
            # Calculate FOV
            FovX = 2 * np.arctan(width / (2 * fx))
            FovY = 2 * np.arctan(height / (2 * fy))

            # Construct image path
            image_path = os.path.join(path, images, f"camera_{camera_name}", f"{frame_id:08d}.jpg")
            # print(f"image_path: {image_path}")
            image_name = f"camera_{camera_name}/{frame_id:08d}.jpg"
            
            # Construct depth map path
            depth_path = ""
            depth_params = None
            if depths:
                depth_path = os.path.join(path, depths, f"camera_{camera_name}", f"{frame_id:08d}.png")
                print(f"depth_path: {depth_path}")
            
            mono_depth_path = ""
            if depths_mono and use_mono_depth:
                mono_depth_path = os.path.join(path, depths_mono, f"camera_{camera_name}", f"{frame_id:08d}.png")
                
            # Determine if the image is a test image
            if eval:
                is_test = frame_id % llffhold == 0
            else:
                is_test = False

            # Construct matches file path
            # Only construct matches path for non-last frames
            matches_path = ""
            matching_feature_points = None
            if use_feature_matching_loss:
                if frame_id < n_frames - 1:
                    # matches path structure: path/matches/camera_XXX/current_frame_next_frame.npy
                    # e.g., ".../matches/camera_BACK/00000000_00000001.npy"
                    matches_path = os.path.join(path, "matches", f"camera_{camera_name}", f"{frame_id:08d}_{frame_id+1:08d}.npy")
                    matching_feature_points = np.load(matches_path, allow_pickle=True).item()

            cam_info = CameraInfo(
                uid=len(cam_infos),
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_params=depth_params,
                image_path=image_path,
                image_name=image_name,
                depth_path=depth_path,
                matches_path = matches_path,
                matching_feature_points=matching_feature_points,
                mono_depth_path=mono_depth_path,
                c2w_nerf=c2w_nerf,
                gt_dof_path="", 
                real_world_scale=1.0,
                width=width,
                height=height,
                is_test=is_test
            )
            cam_infos.append(cam_info)

    # Separate training and testing cameras
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Calculate scene range normalization parameters
    # Load or create point cloud from dataset
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Generate random points using scene range
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # Use scene bounds
        scene_scale = nerf_normalization['radius']
        scene_center = nerf_normalization['translate']
        
        # Generate random point cloud
        xyz = np.random.uniform(-scene_scale, scene_scale, (num_pts, 3)) + scene_center
        rgb = np.random.uniform(0, 1, (num_pts, 3)) * 255
        
        # Save point cloud
        storePly(ply_path, xyz, rgb.astype(np.uint8))

    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
        
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    
    return scene_info


def get_numeric_id(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else 0


def readColmapCameras_our(cam_extrinsics, cam_intrinsics, images_folder, test_cam_names_list, 
                          gt_dof_folder="", mode="no_dof", use_feature_matching_loss=False, 
                          depths_mono_folder="", db_path=None
                          ):
    """
    Read camera information for our custom dataset with strict file existence checking
    
    Args:
        mode: One of ["no_dof", "both"]
        use_feature_matching_loss: Whether to use feature matching loss
    """
    cam_infos = []
    
    
    # Construct full file path mapping for images
    existing_images = {}
    for filename in os.listdir(images_folder):
        if filename.endswith(('.jpg', '.JPG', '.png', '.PNG')):
            full_path = os.path.join(images_folder, filename)
            existing_images[filename] = full_path
    print(f"Found {len(existing_images)} images in images folder")
    
    # Construct full file path mapping for DOF images
    existing_dof_images = {}
    if gt_dof_folder and os.path.exists(gt_dof_folder):
        for filename in os.listdir(gt_dof_folder):
            if filename.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                full_path = os.path.join(gt_dof_folder, filename)
                existing_dof_images[filename] = full_path
        print(f"Found {len(existing_dof_images)} DOF images in {gt_dof_folder}")

    # Read COLMAP database matching information
    adjacency = None
    if use_feature_matching_loss and db_path and os.path.exists(db_path):
        try:
            adjacency = read_colmap_adjacency(db_path)
            print(f"Read {len(adjacency)} image adjacency entries for camera matching")
        except Exception as e:
            print(f"Error reading COLMAP database: {e}")

    # Collect and validate valid cameras
    valid_cameras = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        image_name = extr.name
        
        # Skip camera records without corresponding actual image files
        if (image_name not in existing_images) and (image_name not in existing_dof_images):
            continue   
            
        # Mark whether it is a DOF image and save the corresponding path
        is_dof = image_name in existing_dof_images
        if is_dof:
            image_full_path = existing_dof_images[image_name]
        else:
            image_full_path = existing_images[image_name]
        
        # Save path while collecting camera information
        camera_info = {
            'key': key, 
            'is_dof': is_dof,
            'image_name': image_name,
            'image_path': image_full_path
        }
        valid_cameras.append(camera_info)
    
    print(f"Total cameras: {len(valid_cameras)}")

    # Sort by image name
    # sorted_cameras = sorted(valid_cameras, key=lambda x: cam_extrinsics[x['key']].name)
    sorted_cameras = sorted(valid_cameras, key=lambda x: get_numeric_id(cam_extrinsics[x['key']].name))

    # # Read scale
    # scale_file = os.path.join(os.path.dirname(images_folder), "scale.txt")
    # real_world_scale = 1.0
    # if os.path.exists(scale_file):
    #     with open(scale_file, 'r') as f:
    #         real_world_scale = float(f.read().strip())
    # print(f"Real world scale: {real_world_scale}")
    
    cam_dist = 41.4    # Actual distance between two boundary cameras during shooting, 41.4
    first_cam, last_cam = sorted_cameras[0], sorted_cameras[-1]
    first = cam_extrinsics[first_cam['key']]  # Directly get camera parameter object 
    last = cam_extrinsics[last_cam['key']]    # Same as above
    print(f"first camera: {first.name}\tlast camera: {last.name}")
    # Directly use the tvec attribute of the object
    pose_dist = np.array(first.tvec) - np.array(last.tvec)
    real_world_scale = cam_dist / np.linalg.norm(pose_dist)
    print("real_world_scale is: ", real_world_scale)
    
    # Filter cameras based on mode
    print(f"mode is : {mode}")
    if mode == "no_dof":
        filtered_cameras = [cam for cam in sorted_cameras if not cam['is_dof']]
    else:  # "both"
        dof_train_cameras = [cam for cam in sorted_cameras if cam['is_dof']]
        non_dof_cameras = [cam for cam in sorted_cameras if not cam['is_dof']]
        
        print(f"Total DOF cameras: {len(dof_train_cameras)}")
        print(f"Total non-DOF cameras: {len(non_dof_cameras)}")
        
        # Process non-DOF images
        train_non_dof = []
        test_non_dof = []
        for cam in non_dof_cameras:
            if cam_extrinsics[cam['key']].name in test_cam_names_list:
                test_non_dof.append(cam)
            else:
                train_non_dof.append(cam)
        
        filtered_cameras = dof_train_cameras + train_non_dof + test_non_dof

    # Use saved paths directly when using camera parameters
    for idx, cam in enumerate(filtered_cameras):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(filtered_cameras)))
        sys.stdout.flush()
        
        key = cam['key']
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        # Basic camera parameters
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        # Convert camera pose to NeRF format
        c2w_nerf = colmap_to_nerf_transform(R, T)
        
        # Calculate FoV
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Only undistorted datasets supported!"
            
        # use the pre-saved path
        image_path = cam['image_path']
                
        # Handle feature matching
        matches_path = ""
        matching_feature_points = None
        if use_feature_matching_loss and not cam['is_dof']:
            if adjacency and extr.name in adjacency:
                # Find the best match based on the COLMAP database.  
                best_match = None
                best_match_count = 0
                
                for adj_img_name, match_count in adjacency[extr.name]:
                    # Exclude test set images and DOF images
                    if adj_img_name not in test_cam_names_list and adj_img_name in existing_images:
                        if match_count > best_match_count:
                            best_match = adj_img_name
                            best_match_count = match_count
                
                if best_match:
                    current_name = os.path.splitext(extr.name)[0]
                    next_name = os.path.splitext(best_match)[0]
                    
                    matches_base = os.path.join(os.path.dirname(images_folder), "matches")
                    matches_name = f"{current_name}_{next_name}.npy"
                    matches_path = os.path.join(matches_base, matches_name)
                    
                    if os.path.exists(matches_path):
                        matching_feature_points = np.load(matches_path, allow_pickle=True).item()
            
            # Fallback to index-based matching logic
            elif idx < len(filtered_cameras) - 1:
                # Ensure that both the current frame and the next frame are non-depth maps.
                next_cam = filtered_cameras[idx + 1]
                if not next_cam['is_dof']:  
                    next_key = next_cam['key']
                    next_extr = cam_extrinsics[next_key]
                    
                    current_name = extr.name[:-4]
                    next_name = next_extr.name[:-4]
                    
                    matches_base = os.path.join(os.path.dirname(images_folder), "matches")
                    print(f"matches_base: {matches_base}")
                    matches_subdir = os.path.dirname(os.path.relpath(image_path, images_folder))
                    print(f"matches_subdir: {matches_subdir}")
                    matches_name = f"{current_name}_{next_name}.npy"
                    print(f"matches_name: {matches_name}")
                    matches_path = os.path.join(matches_base, matches_subdir, matches_name)
                    print(f"matches_path: {matches_path}")
                    # exit()
                    
                    if os.path.exists(matches_path):
                        matching_feature_points = np.load(matches_path, allow_pickle=True).item()
        
        image_name = extr.name
        assert image_name == os.path.basename(image_path), f"{image_name} != {os.path.basename(image_path)}"
        
        # Determine if it is a test set
        is_test = False
        if not cam['is_dof']:  # Only non-depth images can be test sets
            is_test = extr.name in test_cam_names_list
                    
        # Other parameter settings
        gt_dof_path = image_path if cam['is_dof'] else ""
        
        base_name = os.path.splitext(extr.name)[0]
        # mono_depth_path = os.path.join(depths_mono_folder, f"{base_name}.png") if depths_mono_folder != "" else ""
        mono_depth_path = ""
        if depths_mono_folder and os.path.exists(depths_mono_folder):
            standard_path = os.path.join(depths_mono_folder, f"{base_name}.png")
            if os.path.exists(standard_path):
                mono_depth_path = standard_path

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None,
                              image_path=image_path, image_name=image_name, depth_path="",
                              matches_path=matches_path, 
                              matching_feature_points=matching_feature_points,
                              mono_depth_path=mono_depth_path, 
                              c2w_nerf=c2w_nerf,
                              gt_dof_path=gt_dof_path, 
                              real_world_scale=real_world_scale,
                              width=width, height=height, is_test=is_test)

        cam_infos.append(cam_info)
    
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo_our(path, images, eval, train_test_exp, llffhold = 8, mode="no_dof", use_feature_matching_loss=False, depths_mono="", use_mono_depth=False):
    """
    Read scene information for our custom dataset
    
    Args:
        mode: One of ["no_dof", "both"]
        use_feature_matching_loss: Whether to use feature matching loss
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse_dof/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse_dof/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse_dof/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse_dof/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    print(f"dof_images: {os.path.join(path, 'dof_images')}")
    gt_dof_folder = os.path.join(path, "dof_images") if (os.path.exists(os.path.join(path, "dof_images")) and mode == "both") else ""
    print(f"gt_dof_folder: {gt_dof_folder}")

    db_path = find_colmap_database(path)
    if db_path:
        print(f"Found COLMAP database for feature matching: {db_path}")
    else:
        print(f"No COLMAP database found, will use sequential frame matching")

    # Handle eval and test set splitting
    test_cam_names_list = []
    if eval:
        # First get the actual images present in the images folder
        images_folder = os.path.join(path, "images" if images is None else images)
        actual_images = set()
        for filename in os.listdir(images_folder):
            if filename.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                actual_images.add(filename)
        
        # Get DOF images from the dof_images folder
        dof_images = set()
        if os.path.exists(gt_dof_folder):
            for filename in os.listdir(gt_dof_folder):
                if filename.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                    dof_images.add(filename)
        
        # Collect only those images that are in the actual images folder but not in the depth of field images folder
        non_dof_images = sorted([img for img in actual_images if img not in dof_images])
        
        # Select non-depth of field images as the test set 
        llffhold = 8
        test_cam_names_list = [name for idx, name in enumerate(non_dof_images) if idx % llffhold == 0]
        print(f"Selected {len(test_cam_names_list)} test images using llffhold={llffhold}")        

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras_our(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        test_cam_names_list=test_cam_names_list,
        gt_dof_folder=gt_dof_folder,
        mode=mode,
        use_feature_matching_loss=use_feature_matching_loss,
        depths_mono_folder=os.path.join(path, depths_mono) if depths_mono != "" and use_mono_depth else "",
        db_path=db_path
    )
    
    # Maintain the original sorting logic
    if use_feature_matching_loss:
        cam_infos = cam_infos_unsorted  # already sorted by name
    else:
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    
    # Validate dataset statistics
    train_dof = sum(1 for c in train_cam_infos if c.gt_dof_path)
    train_no_dof = len(train_cam_infos) - train_dof
    test_no_dof = len(test_cam_infos)
    print(f"Training set: {train_dof} DOF images, {train_no_dof} non-DOF images")
    print(f"Test set: {test_no_dof} non-DOF images")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # Handle point cloud
    ply_path = os.path.join(path, "sparse_dof/0/points3D.ply")
    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse_dof/0/points3D.bin"))
        except:
            xyz, rgb, _ = read_points3D_text(os.path.join(path, "sparse_dof/0/points3D.txt"))
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Auto": readAutoSceneInfo,  # Add new dataset type
    "Custom_DOF": readColmapSceneInfo_our,  # Add new dataset type
}