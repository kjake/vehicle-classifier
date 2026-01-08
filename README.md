# Vehicle Classification Model

This vehicle classification model uses the DrBimmer vehicle classification dataset hosted on Hugging Face. This project uses an augmented version of the dataset, and the label set differs from the original.

https://huggingface.co/datasets/DrBimmer/vehicle-classification

* vehicle_dataset.py - Download the dataset from Hugging Face and export it to an ImageFolder layout for PyTorch.
* train.py - ResNet50 trainer with optional fine-tuning, MixUp/CutMix, schedulers, and metrics export.
* infer.py - Inference for a single image or a directory of images with optional JSON output.
* export.py - Export a trained model to OpenVINO, CoreML, ONNX, and NCNN.

## Class map (augmented dataset)

This repository uses the following 7-class label set (the `classmap.json` included with the augmented dataset):

```json
{
  "0": "Bus",
  "1": "Car",
  "2": "Jeep",
  "3": "Motorcycle",
  "4": "SUV",
  "5": "Truck",
  "6": "Van"
}
```

Ensure you use the matching `classmap.json` for the checkpoint you are running or exporting.

## Quickstart

0. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Create the dataset on disk:

```bash
python vehicle_dataset.py --output-dir vehicle-dataset
```

2. Train the model (basic):

```bash
python train.py --data-dir vehicle-dataset
```

Optional training examples:

```bash
# Resume from a checkpoint, use cosine LR, and write metrics
python train.py --data-dir vehicle-dataset \
  --resume checkpoints/epoch_05.pth \
  --scheduler cosine \
  --write-metrics --metrics-out checkpoints/metrics.json

# Enable MixUp + CutMix with a preset, unfreeze all layers, and use larger inputs
python train.py --data-dir vehicle-dataset \
  --img-size 320 \
  --unfreeze all --unfreeze-epoch 0 \
  --aug mixup0.2+cutmix1.0@p0.6

# Switch device explicitly and enable eval-time TTA
python train.py --data-dir vehicle-dataset --device cuda --tta
```

3. Run inference (single image or directory):

```bash
python infer.py /path/to/image.jpg --checkpoint checkpoints/best_model.pth
python infer.py /path/to/images --checkpoint checkpoints/resnet50_epoch_10.pth --topk 3 --json
```

Inference notes:

* `--classmap` defaults to `./classmap.json`, or `classmap.json` next to the checkpoint if present. With the augmented dataset, use the updated 7-class map shown above (or the one shipped alongside your checkpoint).
* Use `--device auto|cpu|cuda|mps` to select the compute device.
* `--json` emits machine-readable output; `--debug` prints checkpoint metadata.

4. Export model formats:

```bash
python export.py --quiet
python export.py --formats onnx openvino --verbose
```

Export notes:

* `--checkpoint` supports either a raw `state_dict` or a dict containing `model_state_dict`.
* `--input-size` and `--batch` control the dummy input shape for exports.
* Use `--openvino-fp16` or `--openvino-fp32` to override the platform default.
* `--smoke-test` runs a dummy forward pass and prints the top-1 label.

## Notes

The training script expects an ImageFolder layout with `train/` and `test/` splits. If the Hugging Face
dataset only provides a train split, the dataset export script will create a test split automatically.

## Core ML export requirements

Core ML export relies on Core ML Tools and currently works best with:

* Python 3.10–3.12
* PyTorch 2.1–2.3
* coremltools 7.x

Known-good install example:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
