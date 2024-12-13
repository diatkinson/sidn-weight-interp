import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from gnns import GNNForClassification
from jaxtyping import Float, Int
from meta_models.vanilla_transformer import TransformerClassifier
from squeezenet_dataset import get_dataloaders
from torch import Tensor as T


def evaluate(args, model, loader, acc_ks=(1, 5, 10)):
    model.eval()
    total_loss = 0
    total_count = 0
    topk_accs = {k: 0 for k in acc_ks}
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = {k: v.to(args.device).to(args.dtype) for k, v in inputs.items()}
            labels: Float[T, "batch"] = labels.to(args.device).argmax(dim=1)
            outputs: Float[T, "batch 1000"] = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            for k in acc_ks:
                predicted: Int[T, "batch k"] = outputs.topk(k, dim=1, largest=True, sorted=True)[1]
                # Note: won't work when there's more than one correct label (i.e., test_ood=True)
                topk_accs[k] += (predicted == labels.unsqueeze(1)).sum().item()

            total_count += labels.shape[0]

    top_acc = {k: 100 * correct / total_count for k, correct in topk_accs.items()}
    avg_loss = total_loss / (i + 1)  # len(loader)

    return avg_loss, top_acc


def train(args, model, dataloaders, criterion, optimizer, scheduler, max_epochs, patience):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Training {args.model} with {args.dtype} precision")

    for epoch in range(max_epochs):
        total_train_loss = 0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(dataloaders["train"]):
            model.train()
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            labels = labels.to(args.device).argmax(dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / max(i, 1)

            log_status = (i + 1) % 10 == 0
            if log_status:
                val_loss, val_accuracy = evaluate(args, model, dataloaders["val"])
                fmt_val_acc = str({k: round(v, 1) for k, v in val_accuracy.items()})
                tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{tm} {i}/{len(dataloaders['train'])} Train Loss: {avg_train_loss:.4f}, ",
                    f"Val Loss: {val_loss:.4f}, Val accs: {fmt_val_acc}, LR: {scheduler.get_last_lr()[0]:.4f}",
                )
                if int(tm[-1]) % 10 == 0:
                    torch.save(model.state_dict(), f"{args.model}_{datetime.now().isoformat()}.pth")
                    print(f"Model saved as '{args.model}_{datetime.now().isoformat()}.pth'")

        tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{tm} Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc (Top 10): {val_accuracy[10]:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy[10]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break

    test_loss, test_acc = evaluate(args, model, dataloaders["test"])

    return best_val_loss, best_val_acc, test_loss, test_acc


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret finetuned squeezenets")
    parser.add_argument("--data_dir", type=str, default="/home/localdia/model_talk/imagenet_classifiers/data/")
    parser.add_argument("--augmentation_factor", type=int, default=400)
    parser.add_argument("--model", type=str, default="transformer")
    args = parser.parse_args()

    # Hyperparameters
    args.device = "cuda"
    max_epochs = 2
    patience = 10
    max_learning_rate = 0.01
    learning_rate = 1e-3
    weight_decay = 1e-3
    split_layer = 10
    augmentation_factor = args.augmentation_factor
    model_dir = os.path.join(args.data_dir, "squeezenet_binary_classifiers")
    dtype = torch.float32

    if args.model == "transformer":
        args.dtype = dtype
        model = (
            TransformerClassifier(
                d_model=1024,
                num_layers=6,
                num_heads=16,
            )
            .to(args.device)
            .to(args.dtype)
        )
        batch_size = 1024
    else:
        args.dtype = torch.bfloat16
        model = GNNForClassification().to(args.device).to(args.dtype)
        batch_size = 2

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):.3e}")

    # Load the dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        model_dir,
        batch_size=batch_size,
        train_workers=10,
        test_workers=1,
        sizes=[0.9, 0.05, 0.05],
        augmentation_factor=augmentation_factor,
        split_layer=split_layer,
        accuracy_cutoff=0.9,
        include_checkpoints=False,
        test_ood=False,
        dtype=args.dtype,
        form="flattened" if isinstance(model, TransformerClassifier) else None,
    )

    print(
        f"{len(train_loader)} training batches, "
        f"{len(val_loader)} validation batches, "
        f"{len(test_loader)} test batches"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay
    )
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        total_steps=max_epochs * steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    # Train the model with early stopping
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    best_val_loss, best_val_acc, test_loss, test_acc = train(
        args, model, dataloaders, criterion, optimizer, scheduler, max_epochs, patience
    )
    print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc}%")
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc}%")

    # Save the trained MLP model
    torch.save(model.state_dict(), f"{args.model}.pth")
    print(f"Model saved as '{args.model}.pth'")
