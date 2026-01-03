import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import json
import multiprocessing as mp

from contextlib import nullcontext

def main():
    
    start_epoch = 0
    pin_memory = False
    
    # Set device
    if torch.cuda.is_available():
        pin_memory=True
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        pin_memory=False
        device = torch.device("mps")
    else:
        pin_memory=False
        device = torch.device("cpu")

    # --- Add this at the top ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default='animal-dataset', help='Path to ImageFolder dataset root')
    parser.add_argument('--debug', action='store_true', help='Print detailed training configuration before training starts')
    args = parser.parse_args()

    # Hyperparameters
    num_workers = min(8, os.cpu_count() or 1)
    num_epochs = 60
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))

    # Model setup
    num_classes = len(train_dataset.classes)
    model = torchvision.models.resnet50(weights='DEFAULT')

    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not args.resume:
        # Optional: freeze all layers except classifier for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # Multi-GPU support
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from checkpoint: {args.resume}, starting at epoch {start_epoch}")
    else:
        print("🟡 Starting fresh training.")

    if args.resume:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5

    # Optional: mixed precision (CUDA only; MPS AMP is limited)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training + Validation loop
    best_val_loss = float('inf')

    _print_config(
        args=args,
        device=device,
        pin_memory=pin_memory,
        use_amp=use_amp,
        num_workers=num_workers,
        num_epochs=num_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=num_classes,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=pin_memory), labels.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()

            # Mixed precision (optional)
            autocast_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
            with autocast_context:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if use_amp:
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
                inputs, labels = inputs.to(device, non_blocking=pin_memory), labels.to(device, non_blocking=pin_memory)
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

def _print_config(*, args, device, pin_memory, use_amp,
                  num_workers, num_epochs, batch_size, weight_decay,
                  train_dataset, val_dataset, num_classes,
                  optimizer, scheduler, start_epoch,
                  train_transform=None, val_transform=None):
    import sys
    import platform
    import json
    import torch
    import torchvision

    effective_lr = optimizer.param_groups[0].get("lr", None)
    resume_str = args.resume if getattr(args, "resume", None) else "none"

    # Always print a concise one-liner (good for logs)
    print(
        "Config: "
        f"device={device.type} pin_memory={pin_memory} amp={use_amp} "
        f"workers={num_workers} batch={batch_size} epochs={num_epochs} "
        f"lr={effective_lr:.6g} wd={weight_decay} "
        f"train={len(train_dataset)} val={len(val_dataset)} classes={num_classes} "
        f"resume={resume_str}"
    )

    if not getattr(args, "debug", False):
        print()
        return

    classes = getattr(train_dataset, "classes", [])
    cfg = {
        "platform": {
            "python": sys.version.split()[0],
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "torch": {
            "torch": torch.__version__,
            "torchvision": getattr(torchvision, "__version__", "unknown"),
            "threads": torch.get_num_threads(),
        },
        "device": {
            "device": str(device),
            "type": device.type,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "pin_memory": bool(pin_memory),
            "amp_enabled": bool(use_amp),
        },
        "data": {
            "data_dir": getattr(args, "data_dir", None),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_classes": num_classes,
            "classes_preview": classes[:min(20, len(classes))],
        },
        "dataloader": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": bool(num_workers and num_workers > 0),
        },
        "optimization": {
            "epochs": num_epochs,
            "start_epoch": start_epoch,
            "lr": effective_lr,
            "weight_decay": weight_decay,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
        },
        "checkpoint": {
            "resume": getattr(args, "resume", None),
            "checkpoint_dir": "checkpoints",
            "best_model_path": "checkpoints/best_model.pth",
        },
    }

    if train_transform is not None:
        cfg["transforms"] = {
            "train": str(train_transform),
            "val": str(val_transform) if val_transform is not None else None,
        }

    print("\n=== Training configuration (debug) ===")
    print(json.dumps(cfg, indent=2))
    print("=== End configuration ===\n")

if __name__ == "__main__":
    import sys, multiprocessing as mp
    if sys.platform in ("darwin", "win32"):
        mp.set_start_method("spawn", force=True)
    main()