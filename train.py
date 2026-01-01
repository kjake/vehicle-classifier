import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Add this at the top ---
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--data-dir', type=str, default='vehicle-dataset', help='Path to ImageFolder dataset root')
args = parser.parse_args()

# Hyperparameters
num_epochs = 30
batch_size = 512
learning_rate = 0.001
weight_decay = 1e-4

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    # transforms.RandomApply([
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    # ], p=0.2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(args.data_dir, 'train'),
    transform=train_transform
)
val_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(args.data_dir, 'test'),
    transform=val_transform
)

# Get the class-to-index mapping
class_to_idx = train_dataset.class_to_idx  # e.g., {'eagle': 0, 'parrot': 1, ...}

# Reverse it for index-to-class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Save as JSON
with open('classmap.json', 'w') as f:
    json.dump(idx_to_class, f, indent=4)

# DataLoaders
num_workers = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

# Model setup
num_classes = len(train_dataset.classes)
model = torchvision.models.resnet50(pretrained=True)

# Replace final FC layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

if not args.resume:
    # Optional: freeze all layers except classifier for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

# Multi-GPU support
model = nn.DataParallel(model)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

if args.resume and os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Resumed from checkpoint: {args.resume}, starting at epoch {start_epoch}")
else:
    print("🟡 Starting fresh training.")

if args.resume:
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-5

# Optional: mixed precision (requires PyTorch >=1.6 and CUDA)
scaler = torch.amp.GradScaler()

# Training + Validation loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    epoch = epoch + start_epoch if args.resume else epoch
    model.train()
    running_train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision (optional)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if True:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_accuracy = 100.0 * correct / total

    # Adjust learning rate if needed
    scheduler.step(val_loss)

    # Print stats
    print(f"Epoch {epoch+1}/{num_epochs} — "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/resnet50_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }, checkpoint_path)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")

print("Training complete.")
