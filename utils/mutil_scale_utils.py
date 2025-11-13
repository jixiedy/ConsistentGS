import re
import os
import json
from PIL import Image
import torch
import numpy as np


def get_frame(image_name):
    """
    from various formats of image names to extract frame numbers
    """
    base_name = os.path.splitext(image_name)[0]
    
    # try to match different formats of numbers
    patterns = [
        r'(\d+)_cam\d+',   # match 0001_cam0 format
        r'frame_(\d+)',    # match frame_0015 format
        r'^(\d+)',         # match formats starting with numbers, e.g., 0009.jpg
        r'DSCF(\d+)'       # match DSCF5858 format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            return int(match.group(1))
    
    return None


def save_frame_json(save_dir, frame_num, FovX, FovY, c2w_nerf):
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy arrays to lists
    if isinstance(c2w_nerf, np.ndarray):
        c2w_nerf = c2w_nerf.tolist()
    elif isinstance(c2w_nerf, torch.Tensor):
        c2w_nerf = c2w_nerf.cpu().numpy().tolist()
        
    data = {
        str(frame_num): {
            "FovX": float(FovX) * 180 / np.pi,
            "FovY": float(FovY) * 180 / np.pi,
            "Extrinsics": c2w_nerf
        }
    }
    
    json_path = os.path.join(save_dir, f"{frame_num}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def merge_json_files(save_dir, output_name="train.json"):
    """Merge all JSON files in the directory"""
    merged_data = {}
    
    # Get all json files and sort by frame number
    json_files = [f for f in os.listdir(save_dir) if f.endswith('.json') and f != output_name]
    json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    # Merge data
    for json_file in json_files:
        with open(os.path.join(save_dir, json_file), 'r') as f:
            data = json.load(f)
            merged_data.update(data)
    
    # Save merged file
    output_path = os.path.join(save_dir, output_name)
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
        
        
def save_image(image, current_frame, save_dir, image_name):
    """
    Save image (RGB or depth)
    """
    if image.max() > 1.0:
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    else:
        image = torch.clamp(image, 0.0, 1.0)

    image = image.squeeze().detach().cpu()
    
    # If it's an RGB image (3 channels), need to adjust the dimension order
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0)
    
    image = image.numpy()
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(os.path.join(save_dir, f"{current_frame}_{image_name}.png"))
    
 
def save_frame_data(viewpoint_cam, image,
                    mono_depth0, current_depth,
                    rendered_image_dof, gt_image_dof,
                    save_dir, save_frame_interval, 
                    max_frames, processed_frames, 
                    iteration):
    """
    Save frame data (JSON, RGB images, and depth maps)
    
    Args:
        viewpoint_cam: Current viewpoint camera object
        mono_depth0: Depth map tensor, can be None
        current_depth: Current depth map tensor
        save_dir: Save directory
        save_frame_interval: Interval between saved frames
        max_frames: Maximum number of frames to save
        processed_frames: Set of processed frames
        iteration: Current iteration number
    
    Returns:
        bool: Whether a new frame was successfully saved
    """
    if len(processed_frames) >= max_frames:
        return False
        
    current_frame = get_frame(viewpoint_cam.image_name)
    if current_frame is None:
        return False

    # Get the minimum processed frame number (if any)
    last_frame = min(processed_frames) if processed_frames else 0
    
    # Ensure the interval with the previous frame meets the requirement
    if processed_frames and (current_frame - last_frame != save_frame_interval):
        return False
        
    if current_frame in processed_frames:  # Avoid duplicate processing
        return False
    
    if mono_depth0 is None:
        return False
        
    processed_frames.add(current_frame)
    
    # Save JSON
    save_frame_json(save_dir, current_frame, viewpoint_cam.FoVx, viewpoint_cam.FoVy, viewpoint_cam.c2w_nerf)

    # Save images
    save_image(viewpoint_cam.original_image, current_frame, save_dir, "gt")    # Save ground truth image
    save_image(image, current_frame, save_dir, "render")                       # Save rendered image
    save_image(rendered_image_dof, current_frame, save_dir, "rendered_dof")    # Save rendered image (with depth of field)
    save_image(gt_image_dof, current_frame, save_dir, "gt_dof")                # Save ground truth image (with depth of field)
    save_image(mono_depth0, current_frame, save_dir, "mon_depth")              # Save monocular depth estimation depth map
    save_image(current_depth, current_frame, save_dir, "render_depth")         # Save rendered depth map
    
    # Attempt to merge JSON files after each save
    if iteration > 1:
        merge_json_files(save_dir)
    
    return True


def cleanup_individual_jsons(save_dir):
    """
    Delete individual frame JSON files, keeping the merged train.json
    
    Args:
        save_dir: Directory containing JSON files
    """
    try:
        # Ensure train.json exists
        train_json_path = os.path.join(save_dir, "train.json")
        if not os.path.exists(train_json_path):
            print("Warning: train.json not found, skipping cleanup")
            return
            
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
        
        # Delete individual frame JSON files
        for json_file in json_files:
            if json_file != "train.json":
                file_path = os.path.join(save_dir, json_file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Failed to delete {json_file}: {str(e)}")
                    
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")