import os
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Use os.path.join to construct the checkpoint file path
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint')
checkpoint_file = os.path.join(checkpoint_dir, 'sam_vit_h_4b8939.pth')
print(f"Checkpoint file path: {checkpoint_file}", flush=True)

# Check if the checkpoint file exists
if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

# Check if the file has read permissions
if not os.access(checkpoint_file, os.R_OK):
    raise PermissionError(f"Read permission denied for file: {checkpoint_file}")

print("Checkpoint file exists and has read permissions.", flush=True)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}", flush=True)

try:
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_file).to(device)
    print("SAM model loaded successfully.", flush=True)
except Exception as e:
    print(f"Error loading SAM model: {e}", flush=True)
    raise

try:
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Mask generator created successfully.", flush=True)
except Exception as e:
    print(f"Error creating mask generator: {e}", flush=True)
    raise

# Load the image using OpenCV
image_path = os.path.join(os.path.dirname(__file__), 'right_inward_01.png')
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file not found or could not be loaded: {image_path}")

# Convert image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

try:
    masks = mask_generator.generate(image_rgb)
    print("Masks generated successfully.", flush=True)
except Exception as e:
    print(f"Error generating masks: {e}", flush=True)
    raise

# Save the masks and overlay images to the output directory
output_dir = os.path.join(os.path.dirname(__file__), 'output_masks')
os.makedirs(output_dir, exist_ok=True)

for i, mask_dict in enumerate(masks):
    # Extract the actual mask data from the 'segmentation' key
    mask = mask_dict.get('segmentation')
    if mask is None:
        print(f"Mask {i} does not contain 'segmentation' key", flush=True)
        continue

    # Ensure mask is in a format that can be saved by OpenCV
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Scale the mask to enhance visibility
    mask_to_save = mask * 255

    # Resize mask to match overlay dimensions if necessary
    overlay = image.copy()
    if mask.shape != overlay.shape[:2]:
        mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save the mask as a PNG file
    mask_path = os.path.join(output_dir, f'mask_{i}.png')
    cv2.imwrite(mask_path, mask_to_save)
    print(f"Mask {i} saved to {mask_path}", flush=True)

    # Create an overlay of the mask on the original image
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand mask dimensions to match overlay
    overlay[mask_expanded[:, :, 0] > 0] = (0, 255, 0)  # Highlight mask regions in green
    overlay_path = os.path.join(output_dir, f'overlay_{i}.png')
    cv2.imwrite(overlay_path, overlay)
    print(f"Overlay {i} saved to {overlay_path}", flush=True)