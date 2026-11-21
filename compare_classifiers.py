import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from plant_disease import CLASS_NAMES, IMAGE_SIZE, LeafDataset, SEED


def select_sample_indices(labels, max_per_class):
    indices = np.arange(len(labels))
    if max_per_class is None:
        return indices
    if max_per_class < 10:
        raise ValueError("--max-per-class must be at least 10")

    rng = np.random.default_rng(SEED)
    return np.concatenate([
        rng.choice(
            np.flatnonzero(labels == class_index),
            size=min(max_per_class, np.count_nonzero(labels == class_index)),
            replace=False,
        )
        for class_index in range(len(CLASS_NAMES))
    ])


def extract_features(data, indices, device, batch_size):
    encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device).eval()
    loader = DataLoader(
        Subset(data, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_batches = []
    label_batches = []
    with torch.inference_mode():
        for images, labels in loader:
            feature_batches.append(encoder(images.to(device)).cpu().numpy())
            label_batches.append(labels.numpy())

    return np.concatenate(feature_batches), np.concatenate(label_batches)


def compare_classifiers(data_dir, max_per_class, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = LeafDataset(data_dir, CLASS_NAMES, transform)
    all_labels = np.asarray(dataset.targets)
    sample_indices = select_sample_indices(all_labels, max_per_class)

    train_indices, remaining_indices = train_test_split(
        sample_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=all_labels[sample_indices],
    )
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        random_state=SEED,
        stratify=all_labels[remaining_indices],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = extract_features(
        dataset,
        sample_indices,
        device,
        batch_size,
    )
    feature_by_index = {
        int(index): feature
        for index, feature in zip(sample_indices, features)
    }
    labels_by_index = {
        int(index): int(label)
        for index, label in zip(sample_indices, labels)
    }

    train_features = np.stack([feature_by_index[int(i)] for i in train_indices])
    test_features = np.stack([feature_by_index[int(i)] for i in test_indices])
    train_labels = np.asarray([labels_by_index[int(i)] for i in train_indices])
    test_labels = np.asarray([labels_by_index[int(i)] for i in test_indices])

    classifiers = {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }

    results = {
        "dataset": {
            "sample_images": int(len(sample_indices)),
            "train_images": int(len(train_indices)),
            "validation_images": int(len(validation_indices)),
            "test_images": int(len(test_indices)),
            "feature_dimensions": int(features.shape[1]),
            "class_names": CLASS_NAMES,
        },
        "models": {},
    }
    for name, classifier in classifiers.items():
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        results["models"][name] = {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "macro_f1": float(
                f1_score(
                    test_labels,
                    predictions,
                    labels=list(range(len(CLASS_NAMES))),
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                test_labels,
                predictions,
                labels=list(range(len(CLASS_NAMES))),
            ).tolist(),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SVM, Random Forest, and Decision Tree using frozen "
            "ImageNet ResNet18 image features."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    results = compare_classifiers(
        args.data_dir,
        args.max_per_class,
        args.batch_size,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
