import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Tomato___Early_blight",
    "Potato___Late_blight",
]

IMAGE_SIZE = 224
SEED = 42


class LeafDataset(Dataset):
    def __init__(self, root, class_names, transform):
        image_folder = ImageFolder(root)
        available = set(image_folder.classes)
        missing = set(class_names) - available
        if missing:
            raise ValueError(
                f"Missing class folders: {sorted(missing)}. "
                f"Dataset folders found: {image_folder.classes}"
            )

        class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.samples = [
            (path, class_to_idx[image_folder.classes[old_label]])
            for path, old_label in image_folder.samples
            if image_folder.classes[old_label] in class_to_idx
        ]
        self.targets = [label for _, label in self.samples]
        self.transform = transform

        if not self.samples:
            raise ValueError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def make_model(num_classes, pretrained):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    actual = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = sum(p == y for p, y in zip(predictions, actual)) / len(actual)
    return average_loss, accuracy, predictions, actual


def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_data = LeafDataset(args.data_dir, CLASS_NAMES, train_transform)
    eval_data = LeafDataset(args.data_dir, CLASS_NAMES, eval_transform)

    indices = np.arange(len(train_data))
    labels = np.array(train_data.targets)
    if args.max_per_class is not None:
        if args.max_per_class < 10:
            raise ValueError("--max-per-class must be at least 10")
        rng = np.random.default_rng(SEED)
        indices = np.concatenate([
            rng.choice(
                np.flatnonzero(labels == class_index),
                size=min(args.max_per_class, np.count_nonzero(labels == class_index)),
                replace=False,
            )
            for class_index in range(len(CLASS_NAMES))
        ])
    train_indices, remaining = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=labels[indices]
    )
    val_indices, test_indices = train_test_split(
        remaining,
        test_size=0.50,
        random_state=SEED,
        stratify=labels[remaining],
    )

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(eval_data, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(eval_data, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = make_model(len(CLASS_NAMES), pretrained=True).to(device)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
    )

    best_val_accuracy = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        if args.freeze_backbone:
            model.eval()
            model.fc.train()
        else:
            model.train()
        total_loss = 0.0
        correct = 0

        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels_batch).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)
        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, loss_function, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f}, accuracy {val_accuracy:.3f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {"model_state": model.state_dict(), "classes": CLASS_NAMES},
                output_path,
            )

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    _, test_accuracy, predictions, actual = evaluate(
        model, test_loader, loss_function, device
    )
    print(f"\nTest accuracy: {test_accuracy:.3f}")
    print(classification_report(
        actual,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(actual, predictions))
    print(f"Saved model to: {output_path}")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["classes"]

    model = make_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    with Image.open(args.image) as image:
        image = image.convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    predicted_index = probabilities.argmax().item()
    print(f"Prediction: {classes[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Plant disease image classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--output", type=Path, default=Path("plant_model.pth"))
    train_parser.add_argument("--max-per-class", type=int)
    train_parser.add_argument("--freeze-backbone", action="store_true")

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--model", type=Path, default=Path("plant_model.pth"))

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
