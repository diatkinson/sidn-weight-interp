import argparse
import json
import os
import random
import subprocess
import time
import uuid
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.models.squeezenet import Fire
from tqdm import tqdm

DEVICE = "cuda"

# Load pre-trained SqueezeNet and get its transforms
weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1
SQUEEZENET = models.squeezenet1_1(weights=weights).to("cuda")
preprocess = weights.transforms()

# Freeze SqueezeNet weights
for param in SQUEEZENET.features.parameters():
    param.requires_grad = False

# Define data transforms
DATA_TRANSFORMS = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), preprocess]),
    "val": preprocess,
}


def get_classifier(last_pretrained_layer: int):
    final_conv = nn.Conv2d(512, 2, kernel_size=1)
    classifier = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        Fire(64, 16, 64, 64),
        Fire(128, 16, 64, 64),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        Fire(128, 32, 128, 128),
        Fire(256, 32, 128, 128),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        Fire(256, 48, 192, 192),
        Fire(384, 48, 192, 192),
        Fire(384, 64, 256, 256),
        Fire(512, 64, 256, 256),
        nn.Dropout(p=0.5),
        final_conv,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    for module in classifier:
        if isinstance(module, nn.Conv2d):
            if module is final_conv:
                init.normal_(module.weight, mean=0.0, std=0.01)
            else:
                init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    model = nn.Sequential(
        SQUEEZENET.features[: last_pretrained_layer + 1],
        classifier[last_pretrained_layer + 1 :],
        nn.Flatten(1),
    )
    # Freeze pretrained parameters
    for param in model[0].parameters():
        param.requires_grad = False

    return model


def create_binary_dataset(dataset, target_classes):
    binary_dataset = []
    target_class_idxs = [dataset.class_to_idx[target_class] for target_class in target_classes]

    for idx, (img, label) in enumerate(dataset.samples):
        binary_label = 1 if label in target_class_idxs else 0
        binary_dataset.append((idx, img, binary_label))

    # Balance the dataset
    target_samples = [item for item in binary_dataset if item[2] == 1]
    other_samples = [item for item in binary_dataset if item[2] == 0]
    min_samples = min(len(target_samples), len(other_samples))

    balanced_dataset = target_samples[:min_samples] + random.sample(other_samples, min_samples)
    random.shuffle(balanced_dataset)

    # Extract indices and samples separately
    indices = [item[0] for item in balanced_dataset]
    samples = [(item[1], item[2]) for item in balanced_dataset]

    return Subset(dataset, indices), samples


# Modify the Dataset class to ensure binary labels
class BinaryImageNetSubset(torch.utils.data.Dataset):
    def __init__(self, subset, samples):
        self.subset = subset
        self.samples = samples

    def __getitem__(self, index):
        image, label = self.subset[index]
        return image, self.samples[index][1]  # Return the binary label

    def __len__(self):
        return len(self.subset)


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, metadata):
    best_model_wts = None
    best_acc = 0.0
    early_stop_threshold = 0.95

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val", "test"]}

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        if epoch_acc >= early_stop_threshold:
            return best_model_wts, best_acc, epoch, True
        elif args.checkpoint_interval and epoch % args.checkpoint_interval == 0:
            acc, loss = eval_model(model, dataloaders["test"])

            metadata = {**metadata, "max_epochs": num_epochs, "early_stopped": False, "epochs_trained": epoch + 1}
            save_model(model, test_acc=acc, test_loss=loss, **metadata)

    return best_model_wts, best_acc, num_epochs, False


def get_dataloaders(datasets_dict, target_classes, batch_size, num_workers=1):
    train_subset, train_samples = create_binary_dataset(datasets_dict["train"], target_classes)
    test_subset, test_samples = create_binary_dataset(datasets_dict["val"], target_classes)

    # Create BinaryImageNetSubset instances
    train_dataset = BinaryImageNetSubset(train_subset, train_samples)
    test_dataset = BinaryImageNetSubset(test_subset, test_samples)

    # Split train dataset into train and validation
    train_size = len(train_dataset) - 100
    val_size = 100
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1),
    }

    return dataloaders


def get_best_model(dataloaders, target_classes, *, hparams, split_layer):
    metadata = {
        "target_classes": target_classes,
    }

    # Create the model
    model = get_classifier(split_layer).to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=hparams["base_learning_rate"], weight_decay=1e-4
    )

    # Define the learning rate scheduler
    steps_per_epoch = len(dataloaders["train"])
    total_steps = hparams["max_epochs"] * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=hparams["max_learning_rate"],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    # Train the model
    best_model_wts, best_val_acc, epochs_trained, early_stopped = train_model(
        model, criterion, optimizer, scheduler, dataloaders, hparams["max_epochs"], metadata
    )
    metadata = {**metadata, "early_stopped": early_stopped, "epochs_trained": epochs_trained}

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, metadata


def eval_model(model, dataloader):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    running_corrects = 0
    loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    loss = loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)

    return acc.item(), loss


def save_model(model, *, target_classes, test_acc, test_loss, epochs_trained, early_stopped, max_epochs):
    model_id = uuid.uuid4().hex
    model_path = os.path.join(args.data_dir, "squeezenet_binary_classifiers", model_id)

    metadata = {
        "target_classes": target_classes,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "epochs_trained": epochs_trained,
        "max_epochs": max_epochs,
        "early_stopped": early_stopped,
        "split_layer": args.split_layer,
        "seed": random_seed,
        "hostname": hostname,
        "date": datetime.now().isoformat(),
        "version": 2,
    }

    # Save only the trainable parameters
    torch.save({name: param for name, param in model.named_parameters() if param.requires_grad}, f"{model_path}.pth")

    with open(f"{model_path}.json", "w") as f:
        json.dump(metadata, f)


def train_and_save_model(datasets_dict, hparams, num_classes, split_layer=10, num_workers=6):
    start_time = time.time()

    target_classes = random.choices(list(datasets_dict["train"].class_to_idx.keys()), k=num_classes)
    target_class_str = "+".join(target_classes)

    dataloaders = get_dataloaders(datasets_dict, target_classes, hparams["batch_size"], num_workers)

    # Train the model
    model, metadata = get_best_model(dataloaders, target_classes, hparams=hparams, split_layer=split_layer)

    # Evaluate on test set
    test_acc, test_loss = eval_model(model, dataloaders["test"])
    save_model(model, test_acc=test_acc, test_loss=test_loss, max_epochs=hparams["max_epochs"], **metadata)

    # Print summary line
    training_time = time.time() - start_time
    current_time = datetime.now().strftime("%H:%M:%S")
    print(
        f"{current_time} | {random_seed:8d} | {target_class_str:18s} | {metadata['epochs_trained']:3d} | {timedelta(seconds=int(training_time))} | {test_acc:.4f}"
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SqueezeNet Binary Classifier Series Training")
    parser.add_argument("--split_layer", type=int, default=10, help="Index of the last pretrained layer (default: 10)")
    parser.add_argument("--num_models", type=int, default=5_000, help="Number of models to train (default: 5,000)")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes models are trained to identify")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="/home/localdia/model_talk/imagenet_classifiers/data/")
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()

    # Set device
    hostname = subprocess.check_output("hostname").decode().strip()
    if "compute" in hostname:
        hostname = "cais"

    # Generate a random seed
    random_seed = random.randint(1, 2**25)
    print(f"Using random seed: {random_seed}")
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Define hyperparameters
    hparams = dict(
        batch_size=256,
        max_epochs=100,
        base_learning_rate=0.001,
        max_learning_rate=0.01,
    )

    # Count the number of trainable parameters
    model = get_classifier(args.split_layer)
    print("First trainable layer:", model[1][0])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # Load ImageNet dataset and create output directory
    imagenet_dir = os.path.join(args.data_dir, "imagenet")
    datasets_dict = {
        x: datasets.ImageFolder(os.path.join(imagenet_dir, x), DATA_TRANSFORMS[x]) for x in ["train", "val"]
    }
    os.makedirs(os.path.join(args.data_dir, "squeezenet_binary_classifiers"), exist_ok=True)

    # Main training loop
    print("\nTraining models:")
    print("Time     | Seed      | Classes            | Eps | Tr Time | Test Acc")
    print("-" * 70)

    for i in range(args.num_models):
        train_and_save_model(
            datasets_dict,
            hparams,
            num_classes=args.num_classes,
            split_layer=args.split_layer,
            num_workers=args.num_workers,
        )

    print("\nTraining complete!")
