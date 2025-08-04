import base64
import numpy as np
import cv2
from cellpose import models

# Load model once (globally)
model = models.CellposeModel(gpu=True)
def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    return image
def run_segmentation(base64_image: str):
    img = decode_base64_image(base64_image)

    # Convert to RGB
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    masks, flows, styles = model.eval(
        img_rgb,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"tile_norm_blocksize": 150}
    )

    polygons = []
    for label_id in range(1, masks.max() + 1):
        binary_mask = (masks == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            poly = contour.squeeze().tolist()
            if isinstance(poly[0], int):
                poly = [poly]
            polygons.append({
                "label_id": label_id,
                "polygon": poly
            })

    return polygons
def run_segmentation_from_bytes(image_bytes: bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

    # Convert to RGB
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    masks, flows, styles = model.eval(
        img_rgb, channels=[0, 1, 2],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"tile_norm_blocksize": 150}
    )

    polygons = []
    for label_id in range(1, masks.max() + 1):
        binary_mask = (masks == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            poly = contour.squeeze().tolist()
            if isinstance(poly[0], int):
                poly = [poly]
            polygons.append({
                "label_id": label_id,
                "polygon": poly
            })

    return polygons


