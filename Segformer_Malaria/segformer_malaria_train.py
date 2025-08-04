import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from dataset import MalariaDataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import os
# Paths
images_dir = "Malaria/train/images"
masks_dir = "Malaria/train/masks"
checkpoint_dir = "Malaria/models"
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = 16
num_epochs = 100
num_classes = 8
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Optimizes performance for fixed image size

# ------------------ Transforms ------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------ Dataset and Loader ------------------
train_dataset = MalariaDataset(images_dir, masks_dir, transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,  # Faster transfer to GPU
    num_workers=4      # Parallel data loading (adjust based on your CPU)
)

# ------------------ Model ------------------
processor = SegformerImageProcessor(do_reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
model = model.to(device)

# ------------------ Loss and Optimizer ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------ Resume from Checkpoint ------------------
start_epoch = 0
checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint_epoch")]
if checkpoints:
    latest_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from checkpoint: {latest_ckpt} at epoch {start_epoch}")

# ------------------ Training Loop ------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

# ------------------ Save Final Model ------------------
final_model_path = os.path.join(checkpoint_dir, "trained_model.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Training complete. Final model saved to: {final_model_path}")
