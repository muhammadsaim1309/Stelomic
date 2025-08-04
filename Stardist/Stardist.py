import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment this line if you are using CPU
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d # Example image from StarDist
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import numpy as np

# Function to segment an image using StarDist and save the output
def stardist_segment_and_save(input_path, output_dir, model):
    # Read and preprocess image
    img = imread(input_path)
    if img.ndim == 3:
        img = rgb2gray(img)  # Convert to grayscale if RGB

    # Predict labels
    labels, _ = model.predict_instances(normalize(img))

    # Render output image
    output_img = render_label(labels, img=img)
    # Convert to uint8 and keep only RGB channels
    output_img = (output_img[..., :3] * 255).astype(np.uint8)

    # Prepare output path
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"stardist_{base_name}")

    # Save output image
    imsave(output_path, output_img)
    print(f"Saved: {output_path}")
    return output_path

# Example usage:
model = StarDist2D.from_pretrained('2D_versatile_fluo') # Load a pre-trained model
input_image = "test_images/0dcca702-a4ef-4fb3-a940-9c0c140b21c7.png" # Example input image path 
output_dir = "output" # Directory to save output images
output_path = stardist_segment_and_save(input_image, output_dir, model)