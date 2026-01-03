from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def pick_device(requested: str) -> torch.device:
    """Select a device based on user request and availability."""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise SystemExit("Requested device=cuda but CUDA is not available.")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise SystemExit("Requested device=mps but MPS is not available.")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classmap(classmap_path: Path) -> Dict[int, str]:
    if not classmap_path.exists():
        raise FileNotFoundError(f"classmap.json not found: {classmap_path}")
    with classmap_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Training code saves idx_to_class with string keys. Normalize to int keys.
    idx_to_class: Dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                idx_to_class[int(k)] = str(v)
            except Exception as e:
                raise ValueError(f"Invalid classmap key {k!r} in {classmap_path}") from e
    else:
        raise ValueError(f"classmap.json must be a JSON object mapping index->class. Got: {type(raw)}")

    if not idx_to_class:
        raise ValueError(f"classmap.json appears empty: {classmap_path}")

    return idx_to_class


def build_transform() -> transforms.Compose:
    # Match val_transform from training: Resize(256) -> CenterCrop(224) -> Normalize(ImageNet)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def create_model(num_classes: int) -> torch.nn.Module:
    # Use weights=None to avoid any external downloads during inference.
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")  # load on CPU for portability
    # Training script saved either:
    #  - dict with 'model_state_dict' (epoch checkpoints)
    #  - raw state_dict (best_model.pth)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # This is likely a raw state_dict
        state = ckpt
        meta = {}
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    return {"state_dict": state, "meta": meta}


def iter_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image path not found: {path}")

    images: List[Path] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    if not images:
        raise FileNotFoundError(f"No images found under directory: {path}")
    return images


def predict_one(
    model: torch.nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    idx_to_class: Dict[int, str],
    topk: int,
) -> List[Tuple[str, float, int]]:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    x = transform(img).unsqueeze(0)

    # pin_memory is only beneficial for CUDA; keep false by default.
    x = x.to(device, non_blocking=False)

    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits[0], dim=0)
        k = min(topk, probs.numel())
        top_probs, top_idx = torch.topk(probs, k)

    results: List[Tuple[str, float, int]] = []
    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
        label = idx_to_class.get(int(idx), f"<unknown:{idx}>")
        results.append((label, float(prob), int(idx)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with ResNet50 on image(s).")
    parser.add_argument("image_path", type=str, help="Path to an image file, or a directory of images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--classmap",
        type=str,
        default=None,
        help="Path to classmap.json (defaults to ./classmap.json, or alongside the checkpoint if present).",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to print (default: 5).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device selection (default: auto).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text output.")
    parser.add_argument("--debug", action="store_true", help="Print extra diagnostics (device, checkpoint meta).")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    image_path = Path(args.image_path).expanduser().resolve()

    device = pick_device(args.device)

    # Resolve classmap path
    if args.classmap:
        classmap_path = Path(args.classmap).expanduser().resolve()
    else:
        # Prefer classmap next to checkpoint if it exists; otherwise current working directory.
        candidate = checkpoint_path.parent / "classmap.json"
        classmap_path = candidate if candidate.exists() else Path("classmap.json").resolve()

    idx_to_class = load_classmap(classmap_path)
    num_classes = len(idx_to_class)

    model = create_model(num_classes=num_classes)
    ck = load_checkpoint(checkpoint_path, device=device)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)

    model.to(device)
    model.eval()

    transform = build_transform()
    images = iter_images(image_path)

    if args.debug:
        meta = ck.get("meta") or {}
        print("Debug:")
        print(f"  device: {device}")
        print(f"  checkpoint: {checkpoint_path}")
        if meta:
            # Avoid dumping optimizer tensors; training meta is safe.
            safe_meta = {k: meta[k] for k in meta.keys() if k in {"epoch", "train_loss", "val_loss", "val_accuracy"}}
            if safe_meta:
                print(f"  checkpoint_meta: {safe_meta}")
        if missing:
            print(f"  WARNING: missing keys when loading state_dict: {len(missing)}")
        if unexpected:
            print(f"  WARNING: unexpected keys when loading state_dict: {len(unexpected)}")
        print(f"  classmap: {classmap_path} ({num_classes} classes)")
        print(f"  images: {len(images)}")
        print()

    all_results = {}
    for p in images:
        preds = predict_one(
            model=model,
            image_path=p,
            transform=transform,
            device=device,
            idx_to_class=idx_to_class,
            topk=args.topk,
        )
        all_results[str(p)] = [{"class": c, "prob": prob, "index": idx} for c, prob, idx in preds]

        if not args.json:
            print(p)
            for i, (c, prob, idx) in enumerate(preds, start=1):
                print(f"  {i:>2}: {c}  {prob*100:6.2f}%  (idx={idx})")
            print()

    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
