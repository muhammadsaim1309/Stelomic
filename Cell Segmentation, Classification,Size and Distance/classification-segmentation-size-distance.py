from scipy.spatial.distance import cdist
import torch
import matplotlib.pyplot as plt
from skimage import io
from cellpose import utils, models
import numpy as np
import os
from glob import glob
from itertools import chain
from scipy.spatial.distance import cdist
from matplotlib.cm import get_cmap
# --- Run Cellpose ---
def run_cellpose(img, model, flow_threshold=0.4, cellprob_threshold=0.0, tile_norm_blocksize=150):
    masks, flows, styles = model.eval(
        img,
        batch_size=32,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize}
    )
    return masks

# --- Run YOLO ---
def run_yolo(img_path, yolo_model, conf_thresh=0.25):
    results = yolo_model(img_path)
    predictions = results.pred[0]
    class_names = results.names
    labels_with_coords = []

    for *box, conf, cls in predictions:
        if conf < conf_thresh:
            continue
        cls = int(cls.item())
        label = class_names[cls]
        if label.lower() == "red blood cell":
            label = "RBC"  # Shorten label

        x1, y1, x2, y2 = [int(i.item()) for i in box]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        labels_with_coords.append((label, cx, cy))

    return labels_with_coords

# --- Compute centroid for a mask ---
def mask_centroids(masks):
    centroids = []
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background

    for obj_id in unique_ids:
        ys, xs = np.where(masks == obj_id)
        if len(xs) == 0 or len(ys) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        centroids.append((obj_id, cx, cy))

    return centroids

# --- Measure Cell Sizes ---
def measure_cell_sizes(masks, micron_per_pixel=1.0):
    cell_sizes = {}  # {cell_id: area_in_microns}
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background

    for obj_id in unique_ids:
        area_pixels = np.sum(masks == obj_id)
        area_microns = area_pixels * (micron_per_pixel ** 2)
        cell_sizes[obj_id] = area_microns

    return cell_sizes
def compute_centroid_distances(centroids, micron_per_pixel=1.0):
    ids = [cid for cid, _, _ in centroids]
    coords = np.array([[x, y] for _, x, y in centroids])
    distances = cdist(coords, coords) * micron_per_pixel
    return ids, coords, distances

# --- Display + Save ---
def display_and_save_combined(image_path, cellpose_model, yolo_model, save_path):
    img = io.imread(image_path)
    masks = run_cellpose(img, cellpose_model)
    yolo_labels = run_yolo(image_path, yolo_model)

    if isinstance(masks, (list, tuple)) and len(masks) > 1:
        masks = masks[0]
    elif isinstance(masks, np.ndarray) and masks.ndim == 3:
        masks = masks[0]

    outlines = utils.outlines_list(masks)
    centroids = mask_centroids(masks)
    cell_sizes = measure_cell_sizes(masks, micron_per_pixel=1.0)
    ids, coords, distances = compute_centroid_distances(centroids)

    seg_points = np.array([[cx, cy] for _, cx, cy in centroids])
    label_points = np.array([[x, y] for _, x, y in yolo_labels])
    label_texts = [lbl for lbl, _, _ in yolo_labels]

    assigned_labels = {}

    if len(seg_points) > 0 and len(label_points) > 0:
        dists = cdist(label_points, seg_points)
        matched_indices = np.argmin(dists, axis=1)

        used_segments = set()
        for i, seg_idx in enumerate(matched_indices):
            seg_id, cx, cy = centroids[seg_idx]
            if seg_id in used_segments:
                continue
            assigned_labels[seg_id] = label_texts[i]
            used_segments.add(seg_id)

    for seg_id, _, _ in centroids:
        if seg_id not in assigned_labels:
            assigned_labels[seg_id] = "RBC"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)

   
    cmap = get_cmap('tab20')

    for i, outline in enumerate(outlines):
        outline = np.array(outline)
        seg_id, cx, cy = centroids[i]
        label = assigned_labels.get(seg_id, "RBC")
        size = cell_sizes.get(seg_id, 0.0)
        color = cmap(i % 20)

        ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.5)
        ax.text(cx, cy, f"{label}\n{size:.1f} µm²", color='white', fontsize=6,
                ha='center', va='center',
                bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.3'))

    # --- Draw all pairwise distances ---
    for i in range(len(centroids)):
        id1, x1, y1 = centroids[i]
        for j in range(i + 1, len(centroids)):
            id2, x2, y2 = centroids[j]
            dist = distances[i][j]

            ax.plot([x1, x2], [y1, y2], color='white', linestyle='dotted', linewidth=0.6)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            ax.text(mid_x, mid_y, f"{dist:.1f} µm", color='cyan', fontsize=4,
                    ha='center', bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2'))

    ax.axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved image to: {save_path}")


# --- Batch Processing ---
def process_folder(input_folder, output_folder, cellpose_model, yolo_model):
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    image_paths = sorted(chain.from_iterable(
        glob(os.path.join(input_folder, ext)) for ext in extensions
    ))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        display_and_save_combined(img_path, cellpose_model, yolo_model, save_path)

# --- Load models ---
yolo_model_path = 'best-cell.pt' # Change this to your YOLO model path
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)
cellpose_model = models.CellposeModel(gpu=True)

# --- Input/Output folders ---
input_folder = '/content/drive/MyDrive/test_images' # Change this to your input folder path
output_folder = '/content/drive/MyDrive/seg_class_size_distance_output' # Change this to your output folder path

# --- Run ---
process_folder(input_folder, output_folder, cellpose_model, yolo_model)
