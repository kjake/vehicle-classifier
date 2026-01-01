# Vehicle Classification Model

This vehicle classification model uses the DrBimmer vehicle classification dataset hosted on Hugging Face.

https://huggingface.co/datasets/DrBimmer/vehicle-classification

* vehicle_dataset.py - Download the dataset from Hugging Face and export it to an ImageFolder layout for PyTorch.
* train.py - Simple ResNet50 trainer. Supports resume.
* infer.py - Infer a single image given a checkpoint.
* export.py - Export a model to OpenVINO, CoreML, ONNX, and NCNN.

## Quickstart

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
python infer.py path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

## Notes

The training script expects an ImageFolder layout with `train/` and `test/` splits. If the Hugging Face
Dataset only provides a train split, the export script will create a test split automatically.
