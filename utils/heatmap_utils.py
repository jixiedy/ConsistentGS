import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_heatmap_matplotlib(tensor, save_path, cmap='jet', normalize=True, title=None):
    # Ensure tensor is on CPU and convert to numpy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.detach().numpy()

    # Ensure tensor is 2D
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, but got array with shape {array.shape}")

    # Handle NaN and Inf values
    array = np.nan_to_num(array)

    # Normalize to [0,1] range
    if normalize:
        if array.max() > 1.0:
            array = (array - array.min()) / (array.max() - array.min() + 1e-6)
        else:
            array = np.clip(array, 0.0, 1.0)

    # Create image
    array = array * 255
    plt.figure(figsize=(10, 8))
    plt.imshow(array, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()

    if title:
        plt.title(title)

    # Save image
    plt.savefig(save_path)
    plt.close()


def save_heatmap_cv2(tensor, save_path, colormap=cv2.COLORMAP_JET, normalize=True):
    # Convert to numpy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.detach().numpy()
    
    # Remove singleton dimensions
    array = np.squeeze(array)
    # Handle NaN and Inf values
    array = np.nan_to_num(array)
    
    # Normalize to [0, 255]
    if normalize:
        if array.max() > 1.0:
            array = (array - array.min()) / (array.max() - array.min() + 1e-6)
        else:
            array = np.clip(array, 0.0, 1.0)
    
    # Convert to uint8
    array = array * 255
    array = array.astype(np.uint8)
    
    # Check array shape
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array for heatmap, but got array with shape {array.shape}")
    
    # Apply color map
    heatmap = cv2.applyColorMap(array, colormap)
    
    # Save the image
    cv2.imwrite(save_path, heatmap)


def save_heatmap_pil(tensor, save_path, colormap='jet', normalize=True):
    # Convert to numpy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.detach().numpy()

    # Ensure tensor is 2D
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, but got array with shape {array.shape}")
    # Handle NaN and Inf values
    array = np.nan_to_num(array)

    # Normalize to [0,1] range
    if normalize:
        if array.max() > 1.0:
            array = (array - array.min()) / (array.max() - array.min() + 1e-6)
        else:
            array = np.clip(array, 0.0, 1.0)
    
    # Clip values to [0,1]
    array = array * 255
    array = np.clip(array, 0, 1)

    # Get colormap function
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap
    colored_array = cmap(array)  # Returns RGBA values in range [0,1]
    
    # Convert to RGB and scale to [0,255]
    rgb_array = (colored_array[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb_array)
    img.save(save_path)


def blend_heatmap_with_image(tensor, image_tensor, save_path, alpha=0.7, colormap='jet'):
    # Convert tensors to numpy arrays
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    heatmap_array = tensor.detach().numpy()
    image_array = image_tensor.permute(1, 2, 0).detach().numpy()  # (H, W, C)

    # Ensure arrays have correct dimensions
    heatmap_array = np.squeeze(heatmap_array)
    if heatmap_array.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, but got array with shape {heatmap_array.shape}")
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected image array with shape (H, W, 3), but got array with shape {image_array.shape}")

    # Handle NaN and Inf values
    heatmap_array = np.nan_to_num(heatmap_array)
    image_array = np.nan_to_num(image_array)

    # Normalize heatmap to [0,1]
    if heatmap_array.max() != 0:
        heatmap_array = (heatmap_array - heatmap_array.min()) / (heatmap_array.max() - heatmap_array.min() + 1e-6)
    else:
        heatmap_array = np.zeros_like(heatmap_array)
    heatmap_array = np.clip(heatmap_array, 0, 1)

    # Normalize image to [0,1]
    if image_array.max() != 0:
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-6)
    else:
        image_array = np.zeros_like(image_array)
    image_array = np.clip(image_array, 0, 1)

    # Create figure and blend images
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.imshow(heatmap_array, cmap=colormap, alpha=alpha)
    plt.colorbar()

    # Save blended image
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    tensor = torch.randn(822, 1236)
    
    save_heatmap_matplotlib(tensor, 'heatmap_matplotlib.png')
    save_heatmap_cv2(tensor, 'heatmap_cv2.png')
    save_heatmap_pil(tensor, 'heatmap_pil.png')
    
    image_tensor = torch.randn(3, 822, 1236)
    blend_heatmap_with_image(tensor, image_tensor, 'heatmap_blended.png')