import argparse
from pathlib import Path

from datasets import ClassLabel, load_dataset


def resolve_label_names(split):
    label_feature = split.features.get("label")
    if isinstance(label_feature, ClassLabel):
        return label_feature.names
    return None


def save_split(split, split_name, output_dir, label_names):
    for idx, example in enumerate(split):
        label = example["label"]
        label_name = label_names[label] if label_names else str(label)
        image = example["image"]
        class_dir = output_dir / split_name / label_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image_path = class_dir / f"{split_name}_{idx:06d}.jpg"
        image.save(image_path)


def export_dataset(output_dir, test_size):
    dataset = load_dataset("DrBimmer/vehicle-classification")

    if "train" not in dataset:
        split_name = next(iter(dataset.keys()))
        dataset = dataset[split_name].train_test_split(test_size=test_size, seed=42)

    train_split = dataset["train"]
    label_names = resolve_label_names(train_split)
    save_split(train_split, "train", output_dir, label_names)

    if "test" in dataset:
        save_split(dataset["test"], "test", output_dir, label_names)
    elif "validation" in dataset:
        save_split(dataset["validation"], "test", output_dir, label_names)
    else:
        split = train_split.train_test_split(test_size=test_size, seed=42)
        save_split(split["test"], "test", output_dir, label_names)


def main():
    parser = argparse.ArgumentParser(description="Export vehicle dataset to ImageFolder format.")
    parser.add_argument("--output-dir", default="vehicle-dataset", help="Output directory for ImageFolder data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size when splitting train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    export_dataset(output_dir, args.test_size)

    print(f"Dataset export complete: {output_dir}")


if __name__ == "__main__":
    main()
