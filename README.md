# Vehicle Classification Model

This vehicle classification model uses the DrBimmer vehicle classification dataset hosted on Hugging Face.

https://huggingface.co/datasets/DrBimmer/vehicle-classification

* vehicle_dataset.py - Download the dataset from Hugging Face and export it to an ImageFolder layout for PyTorch.
* train.py - Simple ResNet50 trainer. Supports resume.
* infer.py - Infer a single image given a checkpoint.
* export.py - Export a model to OpenVINO, CoreML, ONNX, and NCNN.

## Quickstart

0. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Create the dataset on disk:

```bash
python vehicle_dataset.py --output-dir vehicle-dataset
```

2. Train the model:

```bash
python train.py --data-dir vehicle-dataset
```

3. Run inference:

```bash
python infer_portable.py /path/to/image.jpg --checkpoint checkpoints/best_model.pth
python infer_portable.py /path/to/images --checkpoint checkpoints/resnet50_epoch_10.pth --topk 3 --json
```

4. Export model formats:

```bash
python export.py --quiet
python export.py --formats onnx openvino --verbose
```

## Notes

The training script expects an ImageFolder layout with `train/` and `test/` splits. If the Hugging Face
Dataset only provides a train split, the export script will create a test split automatically.

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
