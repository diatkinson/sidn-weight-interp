import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from jaxtyping import Float
from meta_models.vanilla_transformer import TransformerClassifier
from squeezenet_dataset import get_dataloaders
from torch import Tensor
from tqdm import tqdm


class RandomBaseline:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, inputs):
        tens = next(iter(inputs.values()))
        return torch.randn(tens.shape[0], self.num_classes, device=tens.device)

    def eval(self):
        pass


def evaluate(args, model, loader) -> list[float]:
    model.eval()
    both_correct_counts = [0] * 1000
    one_correct_counts = [0] * 1000
    total_count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = {k: v.to(args.device).to(args.dtype) for k, v in inputs.items()}

            # both labels are Float[Tensor, "batch"]
            labels: Float[Tensor, "batch 2"] = labels.to(args.device)
            assert labels.shape[1] == 2
            outputs: Float[Tensor, "batch 1000"] = model(inputs)

            # Get sorted indices of predictions
            _, pred_indices = outputs.sort(descending=True, dim=1)

            # Check if both labels are in top-k predictions
            for k in range(1, 1001):
                top_k = pred_indices[:, :k]
                labels_in_top_k = [(labels[:, i].unsqueeze(1) == top_k).any(dim=1) for i in range(2)]
                both_correct = labels_in_top_k[0] & labels_in_top_k[1]
                one_correct = labels_in_top_k[0] | labels_in_top_k[1]
                both_correct_counts[k - 1] += both_correct.sum().item()
                one_correct_counts[k - 1] += one_correct.sum().item()

            total_count += labels.size(0)

    # Calculate percentages
    both_correct_pct = [100 * count / total_count for count in both_correct_counts]
    one_correct_pct = [100 * count / total_count for count in one_correct_counts]

    return both_correct_pct, one_correct_pct


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuned squeezenets")
    parser.add_argument("--model", type=str, default="transformer.pth")
    parser.add_argument("--data_dir", type=str, default="/home/localdia/model_talk/imagenet_classifiers/data/")
    args = parser.parse_args()

    # Hyperparameters
    args.device = "cuda"
    args.dtype = torch.float32
    split_layer = 10
    batch_size = 256
    model_dir = os.path.join(args.data_dir, "squeezenet_binary_classifiers")

    state_dict = torch.load(args.model, weights_only=False)
    model = TransformerClassifier(
        d_model=1024,
        num_layers=6,
        num_heads=16,
    )
    model.load_state_dict(state_dict)
    model = model.to(args.device).to(args.dtype)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):.3e}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.3e}")

    _, _, test_loader = get_dataloaders(
        model_dir,
        batch_size=batch_size,
        train_workers=10,
        test_workers=1,
        sizes=[0.9, 0.05, 0.05],
        augmentation_factor=1,
        split_layer=split_layer,
        accuracy_cutoff=0.9,
        use_checkpoints=False,
        test_ood="twoclass",
        form="flattened" if isinstance(model, TransformerClassifier) else None,
    )
    print(f"{len(test_loader)} test batches")

    random_both_correct_pct, random_one_correct_pct = evaluate(args, RandomBaseline(1000), test_loader)
    both_correct_pct, one_correct_pct = evaluate(args, model, test_loader)

    # Plot the results
    k_values = range(1, 1001)

    def create_plot(plot_name, xscale="linear"):
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, both_correct_pct, label="Both Correct")
        plt.plot(k_values, one_correct_pct, label="One Correct")
        plt.plot(k_values, random_both_correct_pct, label="Random Both Correct")
        plt.plot(k_values, random_one_correct_pct, label="Random One Correct")
        plt.xlabel("k")
        plt.ylabel("Accuracy (%)")
        plt.title("Top-k Accuracy Comparison")
        plt.legend()
        plt.xscale(xscale)
        plt.grid(True)
        plt.savefig(f"plots/{plot_name}")
        plt.close()
        print(f"Plot saved as 'plots/{plot_name}'")

    # Normal plot
    create_plot("top_k_accuracy_plot.png")

    # Log plot
    create_plot("top_k_accuracy_log_plot.png", xscale="log")

    def print_accuracies(label, accuracies):
        print(f"{label}:")
        for k in [2, 3, 10, 100, 500, 1000]:
            print(f"Top-{k} accuracy: {accuracies[k-1]:.2f}%")
        print()

    print_accuracies("Both correct", both_correct_pct)
    print_accuracies("One correct", one_correct_pct)
    print_accuracies("Random both correct", random_both_correct_pct)
    print_accuracies("Random one correct", random_one_correct_pct)
