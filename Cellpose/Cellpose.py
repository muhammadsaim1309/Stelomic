import os
from pathlib import Path
import numpy as np
from cellpose import models, core, io, plot
import matplotlib.pyplot as plt
from PIL import Image
def run_cellpose_on_folder(input_folder, output_folder, model, flow_threshold=0.4, cellprob_threshold=0.0, tile_norm_blocksize=0, batch_size=32, channel_indices=[0,1,2]):
    """
    Applies Cellpose to all images in input_folder and saves masks, overlays, and original images as PNG to output_folder.
    Args:
        input_folder (str or Path): Folder with input images (.tif, .png, .jpg).
        output_folder (str or Path): Folder to save results.
        model: Pre-initialized Cellpose model.
        flow_threshold (float): Cellpose flow threshold.
        cellprob_threshold (float): Cellpose cell probability threshold.
        tile_norm_blocksize (int): Block size for normalization.
        batch_size (int): Batch size for Cellpose.
        channel_indices (list): List of channel indices to use.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    files = [f for f in input_folder.iterdir() if f.suffix.lower() in valid_exts]

    if not files:
        print(f'No images found in {input_folder}')
        return

    for img_path in files:
        img = io.imread(str(img_path))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        selected_channels = [i for i in channel_indices if i < img.shape[-1]]
        img_selected = np.zeros_like(img)
        img_selected[:, :, :len(selected_channels)] = img[:, :, selected_channels]

        masks, flows, styles = model.eval(
            img_selected,
            batch_size=batch_size,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize}
        )

        # Save mask as .npy
        # mask_path = output_folder / f"{img_path.stem}_masks.npy"
        # np.save(mask_path, masks)

        # Save overlay image
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img_selected, masks, flows[0])
        # plt.tight_layout()
        # plt.show()
        overlay_path = output_folder / f"{img_path.stem}_overlay.png"
        fig.savefig(overlay_path)

    print(f"All images processed. Results saved to {output_folder}")
run_cellpose_on_folder(
    input_folder="BBBC034",
    output_folder="Cellpose_Results",
    model=models.CellposeModel(gpu=False),
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    tile_norm_blocksize=0,
    batch_size=32,
    channel_indices=[0, 1, 2]
)