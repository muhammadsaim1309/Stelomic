import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def infer_and_plot(image_path, model_weights_path, output_path, num_labels=8):
    # Load input image
    image = Image.open(image_path).convert("RGB")

    # Load processor
    processor = SegformerImageProcessor(do_reduce_labels=False)

    # Load model structure
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device("cpu")))
    model.eval()

    # Preprocess input
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted mask
    pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # Save predicted mask image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_img.save(output_path)

    # Plot original and predicted image
    plt.figure(figsize=(12, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(image)
    plt.title("Original Image")

    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()
infer_and_plot(
    image_path="test/images/0a3b53c7-e7ab-4135-80aa-fd2079d727d6.jpg",
    model_weights_path="trained_model.pt_new",
    output_path="output_masks/pred_0a3b53c7.png"
)
