from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torchvision import models


LOG = logging.getLogger("export")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ExportContext:
    outdir: Path
    input_shape: Tuple[int, int, int, int]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    labels: Dict[str, str]
    model_name: str
    opset: int
    openvino_fp16: bool
    quiet: bool


def _configure_logging(quiet: bool, verbose: bool) -> None:
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _rm_rf(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_config(path: Path, base: Dict[str, Any], files: List[str], fmt: str) -> None:
    cfg = dict(base)
    cfg["format"] = fmt
    cfg["files"] = files
    with (path / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)


def _load_labels(classmap_path: Path) -> Dict[str, str]:
    with classmap_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"classmap must be a dict (got {type(data).__name__})")
    # Keep keys as strings to match your other tooling
    return {str(k): str(v) for k, v in data.items()}


def _build_model(num_classes: int, model_name: str = "resnet50") -> torch.nn.Module:
    if model_name != "resnet50":
        raise ValueError(f"Unsupported model_name={model_name!r} (only 'resnet50' in this script)")
    model = models.resnet50(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_checkpoint_into(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")

    # Handle dict checkpoint that contains model_state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_dict).__name__}")

    # Strip 'module.' prefix (DataParallel) if present
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict

        new_state = OrderedDict()
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith("module."):
                new_state[k.replace("module.", "", 1)] = v
            else:
                new_state[k] = v
        state_dict = new_state

    model.load_state_dict(state_dict)


def _dummy_input(shape: Tuple[int, int, int, int], mean, std) -> torch.Tensor:
    """Create a normalized dummy input matching ImageNet normalization."""
    img = np.zeros(shape, dtype=np.float32)

    mean_np = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std_np = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
    img = ((img - mean_np) / std_np).astype(np.float32)

    return torch.from_numpy(img)


def _run_cmd(cmd: List[str], *, quiet: bool, cwd: Optional[Path] = None) -> None:
    LOG.debug("Running command: %s", " ".join(cmd))
    if quiet:
        # Capture output unless verbose logging is enabled (quiet implies WARN level)
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def export_onnx(ctx: ExportContext, model: torch.nn.Module) -> None:
    path = ctx.outdir / "onnx"
    _rm_rf(path)
    _mkdir(path)

    example = _dummy_input(ctx.input_shape, ctx.mean, ctx.std)

    LOG.info("Exporting ONNX (opset=%d) -> %s", ctx.opset, path / "model.onnx")
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(path / "model.onnx"),
            verbose=False,
            input_names=["input"],
            output_names=["logits"],
            opset_version=ctx.opset,
        )

    base = _base_config(ctx)
    _write_config(path, base, files=["model.onnx"], fmt="onnx")


def export_openvino(ctx: ExportContext) -> None:
    try:
        import openvino as ov  # type: ignore
    except Exception as e:
        raise RuntimeError("openvino is not installed in this environment.") from e

    onnx_path = ctx.outdir / "onnx" / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Run ONNX export first.")

    path = ctx.outdir / "openvino"
    _rm_rf(path)
    _mkdir(path)

    LOG.info("Converting to OpenVINO -> %s", path)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(path / "model.xml"), compress_to_fp16=ctx.openvino_fp16)

    base = _base_config(ctx)
    _write_config(path, base, files=["model.xml", "model.bin"], fmt="openvino")


def export_coreml(ctx: ExportContext, model: torch.nn.Module) -> None:
    # Guardrails to avoid mysterious converter failures
    supported_python_min = (3, 10)
    supported_python_max = (3, 12)
    supported_torch_minors = {(2, 1), (2, 2), (2, 3)}

    if not (supported_python_min <= sys.version_info[:2] <= supported_python_max):
        raise RuntimeError("Core ML export requires Python 3.10–3.12 (recommend 3.11).")

    torch_version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    torch_version = tuple(int(part) for part in torch_version_parts[:2] if part.isdigit())
    if torch_version not in supported_torch_minors:
        raise RuntimeError("Core ML export known-good: PyTorch 2.1–2.3 with coremltools 7.x.")

    try:
        import coremltools as ct  # type: ignore
    except Exception as e:
        raise RuntimeError("coremltools is not installed in this environment.") from e

    path = ctx.outdir / "coreml"
    _rm_rf(path)
    _mkdir(path)

    # CoreML conversion is most reliable from a CPU TorchScript trace.
    example = _dummy_input(ctx.input_shape, ctx.mean, ctx.std)
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=True).eval()

    LOG.info("Converting to Core ML (mlprogram) -> %s", path / "model.mlpackage")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=ctx.input_shape)],
    )
    mlmodel.save(str(path / "model.mlpackage"))

    # Files list is primarily informative; content varies by Core ML tooling/version.
    base = _base_config(ctx)
    _write_config(path, base, files=["model.mlpackage"], fmt="coreml")


def export_ncnn(ctx: ExportContext, model: torch.nn.Module) -> None:
    # Requires pnnx available on PATH
    pnnx = shutil.which("pnnx")
    if not pnnx:
        raise RuntimeError("pnnx was not found on PATH. Install/build pnnx and ensure it is available.")

    path = ctx.outdir / "ncnn"
    _rm_rf(path)
    _mkdir(path)

    example = _dummy_input(ctx.input_shape, ctx.mean, ctx.std)
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=True).eval()
        pt_path = path / "model.pt"
        traced.save(str(pt_path))

    # Convert TorchScript -> NCNN via pnnx
    inputshape = "[" + ",".join(str(x) for x in ctx.input_shape) + "]"
    pt_path_resolved = pt_path.resolve()
    cmd = [pnnx, str(pt_path_resolved), f"inputshape={inputshape}"]
    if not pt_path.exists():
        raise RuntimeError(f"TorchScript file not found: {pt_path}")
    LOG.info("Running pnnx -> %s", path)
    try:
        _run_cmd(cmd, quiet=ctx.quiet, cwd=path)
    except subprocess.CalledProcessError as e:
        # Surface pnnx output when running in --quiet mode
        if hasattr(e, "stdout") and e.stdout:
            LOG.error("pnnx stdout:\n%s", e.stdout.decode(errors="replace"))
        if hasattr(e, "stderr") and e.stderr:
            LOG.error("pnnx stderr:\n%s", e.stderr.decode(errors="replace"))
        raise

    # Verify expected artifacts exist; if pnnx used a different prefix, normalize names.
    param_candidates = list(path.glob("*.ncnn.param"))
    bin_candidates = list(path.glob("*.ncnn.bin"))
    if not param_candidates or not bin_candidates:
        raise RuntimeError("pnnx completed but did not produce *.ncnn.param/*.ncnn.bin outputs")

    param_src = param_candidates[0]
    bin_src = bin_candidates[0]
    param_dst = path / "model.ncnn.param"
    bin_dst = path / "model.ncnn.bin"
    if param_src != param_dst:
        param_src.replace(param_dst)
    if bin_src != bin_dst:
        bin_src.replace(bin_dst)


    # pnnx outputs *.ncnn.param/bin alongside the input model.pt by default.
    input_shape_str = "[" + ",".join(str(x) for x in ctx.input_shape) + "]"
    pt_path_resolved = pt_path.resolve()
    cmd = [pnnx, str(pt_path_resolved), f"inputshape={input_shape_str}"]
    LOG.info("Converting to NCNN via pnnx -> %s", path)
    _run_cmd(cmd, quiet=ctx.quiet, cwd=path)

    base = _base_config(ctx)
    # The usual pnnx outputs
    _write_config(path, base, files=["model.ncnn.param", "model.ncnn.bin"], fmt="ncnn")


def _base_config(ctx: ExportContext) -> Dict[str, Any]:
    return {
        "input_shape": list(ctx.input_shape),
        "model": ctx.model_name,
        "mean": list(ctx.mean),
        "std": list(ctx.std),
        "labels": ctx.labels,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained model into multiple formats.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"),
                   help="Path to checkpoint (raw state_dict or dict w/ model_state_dict).")
    p.add_argument("--classmap", type=Path, default=Path("classmap.json"),
                   help="Path to classmap.json (index->label mapping).")
    p.add_argument("--outdir", type=Path, default=Path("models"),
                   help="Output directory root (formats create subfolders).")
    p.add_argument("--formats", nargs="+", default=["onnx", "openvino", "coreml", "ncnn"],
                   choices=["onnx", "openvino", "coreml", "ncnn"],
                   help="Which formats to export.")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    p.add_argument("--input-size", type=int, default=224, help="Input image size (assumes 3xHxW).")
    p.add_argument("--batch", type=int, default=1, help="Batch size for export dummy input.")
    p.add_argument("--openvino-fp16", action="store_true",
                   help="Force OpenVINO weight compression to FP16 (default: true on non-macOS).")
    p.add_argument("--openvino-fp32", action="store_true",
                   help="Force OpenVINO to keep FP32 weights (default: true on macOS).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a small dummy forward pass and print top-1 label.")
    p.add_argument("--quiet", action="store_true", help="Reduce third-party tool noise.")
    p.add_argument("--verbose", action="store_true", help="More detailed logs.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.quiet, args.verbose)

    labels = _load_labels(args.classmap)
    num_classes = len(labels)

    # Build and load model on CPU for export stability/portability.
    model = _build_model(num_classes=num_classes, model_name="resnet50")
    _load_checkpoint_into(model, args.checkpoint)
    model = model.to("cpu").eval()

    input_shape = (args.batch, 3, args.input_size, args.input_size)

    # Choose OpenVINO fp16 default: keep fp32 on macOS unless overridden.
    if args.openvino_fp16 and args.openvino_fp32:
        raise SystemExit("Choose only one of --openvino-fp16 or --openvino-fp32.")
    if args.openvino_fp16:
        openvino_fp16 = True
    elif args.openvino_fp32:
        openvino_fp16 = False
    else:
        openvino_fp16 = (platform.system() != "Darwin")

    ctx = ExportContext(
        outdir=args.outdir,
        input_shape=input_shape,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        labels=labels,
        model_name="resnet50",
        opset=args.opset,
        openvino_fp16=openvino_fp16,
        quiet=args.quiet,
    )

    LOG.info("Checkpoint: %s", args.checkpoint)
    LOG.info("Outdir: %s", args.outdir)
    LOG.info("Formats: %s", ", ".join(args.formats))
    LOG.info("Input shape: %s", input_shape)

    if args.smoke_test:
        x = _dummy_input(input_shape, ctx.mean, ctx.std)
        with torch.no_grad():
            logits = model(x)[0]
            probs = logits.softmax(dim=0)
            idx = int(torch.argmax(probs).item())
        label = labels.get(str(idx), str(idx))
        LOG.warning("Smoke-test top1: %s (idx=%d)", label, idx)

    # Perform exports (order matters for OpenVINO which uses ONNX)
    fmts: List[str] = list(args.formats)
    if "openvino" in fmts and "onnx" not in fmts:
        LOG.info("OpenVINO requested; ONNX will be exported first as a dependency.")
        fmts = ["onnx"] + fmts

    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for f in fmts:
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    for fmt in ordered:
        try:
            if fmt == "onnx":
                export_onnx(ctx, model)
            elif fmt == "openvino":
                export_openvino(ctx)
            elif fmt == "coreml":
                export_coreml(ctx, model)
            elif fmt == "ncnn":
                export_ncnn(ctx, model)
            else:
                raise AssertionError(fmt)
        except Exception as e:
            LOG.error("Export failed for %s: %s", fmt, e)
            if args.verbose:
                raise
            return 2

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())