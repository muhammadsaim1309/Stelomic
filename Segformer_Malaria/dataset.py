from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms as T

class MalariaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, num_classes=8):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.num_classes = num_classes

        # Only include images that have corresponding masks
        self.masks = sorted(os.listdir(masks_dir))
        self.images = []
        for mask_name in self.masks:
            if os.path.exists(os.path.join(images_dir, mask_name)):
                self.images.append(mask_name)

        # Resize transform for masks only
        self.mask_transform = T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure mask is single-channel

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Apply resize to mask only
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        # Clamp mask values to valid class range [0, num_classes - 1]
        mask = torch.clamp(mask, 0, self.num_classes - 1)

        return image, mask
