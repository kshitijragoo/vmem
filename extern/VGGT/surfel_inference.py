# vmem/extern/VGGT/surfel_inference.py

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torchvision.transforms.functional as F 


# Add vggt to path
vggt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vggt"))
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map




def fov_to_focal_length(fov_rad, image_dim_pixels):
    """
    Converts a field of view in radians to a focal length in pixels.

    Args:
        fov_rad (torch.Tensor): Field of view in radians.
        image_dim_pixels (int): The corresponding image dimension (height or width) in pixels.

    Returns:
        torch.Tensor: The focal length in pixels.
    """
    return (image_dim_pixels / 2.0) / torch.tan(fov_rad / 2.0)



def run_inference_from_pil(
    input_images: List[Image.Image],
    vggt_model: VGGT,
    poses: Optional[np.ndarray] = None,
    depths: Optional[torch.Tensor] = None,
    lr: float = 0.01,
    niter: int = 500,
    visualize: bool = False,
    device: str = 'cuda'
) -> Dict:
    """
    Run VGGT inference on PIL images to extract 3D scene information.
    
    Args:
        input_images: List of PIL images
        vggt_model: VGGT model instance
        poses: Optional camera poses [N, 4, 4]
        depths: Optional depth maps [N, H, W]
        lr: Learning rate for optimization
        niter: Number of optimization iterations
        visualize: Whether to visualize results
        device: Device to run on
        
    Returns:
        Dictionary containing:
        - point_clouds: List of point clouds [N, H, W, 3]
        - confidences: List of confidence maps [N, H, W]
        - depths: List of depth maps [N, H, W]
        - camera_info: Camera information
    """
    
    # Get patch size directly from the VGGT model's architecture
    patch_size = vggt_model.aggregator.patch_embed.patch_size
    patch_H = patch_size
    patch_W = patch_size

    # Convert PIL images to tensors and resize them to be compatible
    images_tensor = []
    for img in input_images:
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        else:
            # Assumes input is already a tensor if not a PIL image
            img_tensor = img

        # Get current dimensions
        _, H, W = img_tensor.shape
        
        # Calculate the nearest dimensions that are a multiple of the patch size
        new_H = round(H / patch_H) * patch_H
        new_W = round(W / patch_W) * patch_W

        # Resize the image if its dimensions are not compatible
        if new_H != H or new_W != W:
            print(f"INFO: Resizing image for VGGT from ({H}, {W}) to ({new_H}, {new_W}) to match patch size.")
            img_tensor = F.resize(img_tensor, [new_H, new_W], antialias=True)

        images_tensor.append(img_tensor)
    
    # Stack images: [N, 3, H, W]
    images_batch = torch.stack(images_tensor).to(device)
    
    # Run VGGT inference
    import time
    print(f"[VGGT Timing] Starting VGGT inference on {len(input_images)} images...")
    print(f"[VGGT Timing] Input batch device: {images_batch.device}, shape: {images_batch.shape}")
    print(f"[VGGT Timing] Model device: {next(vggt_model.parameters()).device}")
    vggt_start = time.time()
    with torch.no_grad():
        outputs = vggt_model(images_batch.unsqueeze(0))  # Add batch dimension
    print(f"[VGGT Timing] VGGT inference completed in {time.time() - vggt_start:.2f}s")
    
    # Extract results
    point_clouds = []
    confidences = []
    depth_maps = []
    
    # Get world points and confidence
    world_points = outputs['world_points'][0]  # Remove batch dimension
    world_points_conf = outputs['world_points_conf'][0]
    depth_pred = outputs['depth'][0]
    
    for i in range(len(input_images)):
        point_clouds.append(world_points[i])
        confidences.append(world_points_conf[i])
        depth_maps.append(depth_pred[i])
    
    # --- START: MODIFIED SECTION ---
    # Extract predicted focal lengths from camera head output
    # pose_enc shape is [S, 9], where the last 2 elements are FoV_y and FoV_x
    pose_enc = outputs['pose_enc'][0] # Remove batch dim, shape is [S, 9]
    fov_y, fov_x = pose_enc[:, 7], pose_enc[:, 8]

    # Convert field of view (in radians) to focal length (in pixels)
    _, _, H, W = images_batch.shape
    focal_y = fov_to_focal_length(fov_y, H)
    focal_x = fov_to_focal_length(fov_x, W)
    
    # Create a list of focal lengths for each frame
    focals_list = torch.stack([focal_x, focal_y], dim=-1).detach().cpu().numpy()

    camera_info = {
        # Return the list of per-frame focal lengths
        'focal': [f for f in focals_list],
        'principal': [0.5, 0.5] # Principal point is assumed to be at the center
    }
    # --- END: MODIFIED SECTION ---
    
    return {
        'point_clouds': point_clouds,
        'confidences': confidences,
        'depths': depth_maps,
        'camera_info': camera_info
    }


def add_path_to_vggt(model_path: str):
    """
    Add VGGT model path to the system.
    This is a placeholder function to match the CUT3R interface.
    """
    pass
