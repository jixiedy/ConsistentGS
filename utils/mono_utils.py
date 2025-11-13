import torch
import gc
import os
from PIL import Image
import numpy as np


def process_mono_depth(cam, mon_model):
    with torch.no_grad():
        input_image = cam.original_image.unsqueeze(0).cuda()
        depth, confidence, output_dict = mon_model.inference({'input': input_image})
        
        del input_image, confidence, output_dict
        torch.cuda.empty_cache()
        gc.collect()
        
        return depth


if __name__ == '__main__':
    # metric3d_vit_small、metric3d_vit_large、metric3d_vit_giant2、metric3d_convnext_large
    check = "metric3d_vit_large"
    mon_model = torch.hub.load('yvanyin/metric3d', check, pretrain=True).cuda()
    mon_model.eval()
    
    root_dirs = ["/data01/dy/datasets/MipNeRF360/bicycle",
                 "/data01/dy/datasets/MipNeRF360/flowers",
                 "/data01/dy/datasets/MipNeRF360/garden",
                 "/data01/dy/datasets/MipNeRF360/stump",
                 "/data01/dy/datasets/MipNeRF360/treehill",
                 "/data01/dy/datasets/MipNeRF360/room",
                 "/data01/dy/datasets/MipNeRF360/counter",
                 "/data01/dy/datasets/MipNeRF360/kitchen",
                 "/data01/dy/datasets/MipNeRF360/bonsai",
                 ]

    resolution = -1
    resolution_scale = 1.0
    for root_dir in root_dirs:
        print(f"starting: {root_dir}")
        image_dir = os.path.join(root_dir, "images")
        suffix = check.split("metric3d_")[-1] 
        mon_depth_dir = os.path.join(root_dir, f"mon_depth_{suffix}")
        os.makedirs(mon_depth_dir, exist_ok=True)

        images = os.listdir(image_dir)

        for image_name in images:
            print(f"image_name: {image_name}")
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert('RGB')

            orig_w, orig_h = image.size
            if resolution in [1, 2, 4, 8]:
                new_size = (round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution)))
            else:  # should be a type that converts to float
                if resolution == -1:
                    print(f"Using default resolution: {orig_w}x{orig_h}")
                    if orig_w > 1600:
                        global_down = orig_w / 1600
                    else:
                        global_down = 1
                else:
                    if isinstance(resolution, (int, float)):
                        global_down = orig_w / resolution
                    else:
                        raise TypeError(f"Resolution should be a number, got {type(resolution)}")
                
                scale = float(global_down) * float(resolution_scale)
                new_size = (int(orig_w / scale), int(orig_h / scale))
            
            image = image.resize(new_size, Image.BILINEAR)

            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0).cuda()
            print(f"image shape: {image.shape}")

            mon_depth, confidence, output_dict = mon_model.inference({'input': image})

            max_depth = mon_depth.max()
            scale_factor = 65535 / max_depth
            mon_depth = (mon_depth * scale_factor).clip(0, 65535).to(torch.uint16)
            mon_depth = mon_depth.squeeze().cpu().numpy()
            print(f"Depth stats - min: {mon_depth.min()}, max: {mon_depth.max()}, mean: {mon_depth.mean()}")
            
            # mon_depth_path = os.path.join(mon_depth_dir, image_name)
            base_name = os.path.splitext(image_name)[0]
            mon_depth_path = os.path.join(mon_depth_dir, f"{base_name}.png")
            Image.fromarray(mon_depth).save(mon_depth_path)
            saved_depth = np.array(Image.open(mon_depth_path))
            print(f"Saved depth stats - min: {saved_depth.min()}, max: {saved_depth.max()}, mean: {saved_depth.mean()}")

            del image, mon_depth, confidence, output_dict
            torch.cuda.empty_cache()
            gc.collect()

        print(f"ending: {root_dir}")
