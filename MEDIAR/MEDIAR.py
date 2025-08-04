import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from train_tools.models import MEDIARFormer
from core.MEDIAR import Predictor

def segment(weights_path, input_folder, output_folder, model_args=None, device="cpu", algo_params={"use_tta": False}):
    """
    Loads weights, uses Predictor.conduct_prediction on all images in input_folder, saves output masks in output_folder.
    Only displays the first image's input/output.
    """
    if model_args is None:
        model_args = {
            "classes": 3,
            "decoder_channels": [1024, 512, 256, 128, 64],
            "decoder_pab_channels": 256,
            "encoder_name": 'mit_b5',
            "in_channels": 3
        }
    if algo_params is None:
        algo_params = {"use_tta": False}

    # Load model and weights
    model = MEDIARFormer(**model_args)
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List images in input folder (jpg, png, tiff)
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    img_files.sort()

    # Use Predictor to process all images
    predictor = Predictor(model, device, input_folder, output_folder, algo_params=algo_params)
    predictor.img_names = img_files
    _ = predictor.conduct_prediction()

    # Show only the first image's input and output
    if img_files:
        first_img = img_files[0]
        input_image_path = os.path.join(input_folder, first_img)
        output_image_path = os.path.join(output_folder, os.path.splitext(first_img)[0] + '_label.tiff')
        img = io.imread(input_image_path)
        pred = io.imread(output_image_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        axes[1].imshow(pred, cmap="cividis")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

# Example usage:
segment("MEDIAR_Weights/from_phase1.pth", "Input_Images_Path/", "Output_Images_Path")