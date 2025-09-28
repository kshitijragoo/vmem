# vmem/extern/VGGT/surfel_inference.py

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image

# Add vggt to path
vggt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vggt"))
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map


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
    # Convert PIL images to tensors
    images_tensor = []
    for img in input_images:
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        else:
            img_tensor = img
        images_tensor.append(img_tensor)
    
    # Stack images: [N, 3, H, W]
    images_batch = torch.stack(images_tensor).to(device)
    
    # Run VGGT inference
    with torch.no_grad():
        outputs = vggt_model(images_batch.unsqueeze(0))  # Add batch dimension
    
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
    
    # Create camera info
    camera_info = {
        'focal': [0.5, 0.5],  # Default focal length
        'principal': [0.5, 0.5]  # Default principal point
    }
    
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
