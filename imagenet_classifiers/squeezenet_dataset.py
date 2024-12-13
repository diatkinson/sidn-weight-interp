import itertools
import json
import math
import os
import random
from functools import partial
from pathlib import Path

import finetune_squeezenets as ft
import torch
import torch.nn.functional as F
from torch.distributions.log_normal import LogNormal
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

DEVICE = "cuda"


def get_full_squeezenet_state_dict(partial_state_dict):
    full_state_dict = partial_state_dict.copy()

    for k, v in ft.SQUEEZENET.state_dict().items():
        layer_parts = k.split(".")
        layer = int(layer_parts[1])
        if layer_parts[0] == "features" and layer < 11:
            layer_name = ".".join(layer_parts[1:])
            full_state_dict[f"0.{layer_name}"] = v.to("cuda")

    return full_state_dict


def get_logits(model, dataloader):
    model.eval()
    logits = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            logits.append(outputs)
    return torch.cat(logits, dim=0)


def test_augmentation(augment_fn, state_dict, labels, imgn_datasets, *, split_layer=10):
    """
    load squeezenet, eval the model, permute the weights, eval the permuted model, verify accuracy is unchanged
    """
    batch_size = state_dict["1_11_squeeze_weight"].shape[0]
    device = state_dict["1_11_squeeze_weight"].device
    class_lookup = {i: c for c, i in imgn_datasets["train"].class_to_idx.items()}

    for b in range(batch_size):
        unbatched_state_dict = {k.replace("_", "."): v[b] for k, v in state_dict.items()}
        unbatched_labels = labels[b]
        target_classes = [class_lookup[label] for label in unbatched_labels.argwhere().flatten().tolist()]

        dataloaders = ft.get_dataloaders(imgn_datasets, target_classes, batch_size=8)

        model = ft.get_classifier(split_layer)
        model.load_state_dict(get_full_squeezenet_state_dict(unbatched_state_dict))
        model.to(device)

        torch.manual_seed(b)
        original_logits = get_logits(model, dataloaders["test"])

        augment_fn(unbatched_state_dict, seed=b)

        model = ft.get_classifier(split_layer)
        model.load_state_dict(get_full_squeezenet_state_dict(unbatched_state_dict))
        model.to(device)

        torch.manual_seed(b)
        augmented_logits = get_logits(model, dataloaders["test"])

        max_diff = (original_logits - augmented_logits).abs().max()
        order_preserved = (original_logits.sort().indices == augmented_logits.sort().indices).all()
        if max_diff > 0.1 or not order_preserved:
            return False
    return True


def permute_and_scale(state_dict, seed, permute=True, scale_factor: int | None = None):
    """
    '1.11.squeeze': torch.Size([64, 384, 1, 1]),
    '1.11.expand1x1': torch.Size([256, 64, 1, 1]),
    '1.11.expand3x3': torch.Size([256, 64, 3, 3]),
    '1.12.squeeze': torch.Size([64, 512, 1, 1]),
    '1.12.expand1x1': torch.Size([256, 64, 1, 1]),
    '1.12.expand3x3': torch.Size([256, 64, 3, 3]),
    '1.14': torch.Size([2, 512, 1, 1]),
    """
    assert state_dict["1.11.squeeze.weight"].shape[0] == 64, "Haven't implemented batched permute and scale yet"
    generator = torch.Generator().manual_seed(seed)

    def pas_expand(squeeze, expand1x1, expand3x3):
        # The outputs of the squeeze layer are sent to both expand layers
        squeeze_weight = state_dict[f"{squeeze}.weight"]
        squeeze_bias = state_dict[f"{squeeze}.bias"]
        expand1x1_weight = state_dict[f"{expand1x1}.weight"]
        expand3x3_weight = state_dict[f"{expand3x3}.weight"]

        channel_dim = squeeze_weight.shape[0]
        perm = torch.randperm(channel_dim, generator=generator) if permute else torch.arange(channel_dim)

        squeeze_weight.data = squeeze_weight.data[perm]
        squeeze_bias.data = squeeze_bias.data[perm]

        expand1x1_weight.data = expand1x1_weight.data[:, perm]
        expand3x3_weight.data = expand3x3_weight.data[:, perm]

    def pas_squeeze(expand1x1, expand3x3, squeeze):
        # The outputs of the two expand layers are cat'd together and fed into the squeeze layer
        expand1x1_weight = state_dict[f"{expand1x1}.weight"]
        expand1x1_bias = state_dict[f"{expand1x1}.bias"]
        expand3x3_weight = state_dict[f"{expand3x3}.weight"]
        expand3x3_bias = state_dict[f"{expand3x3}.bias"]
        squeeze_weight = state_dict[f"{squeeze}.weight"]
        squeeze_bias = state_dict[f"{squeeze}.bias"]

        channel_dim = expand1x1_weight.shape[0]
        assert expand3x3_weight.shape[0] == channel_dim

        expand1x1_perm = torch.randperm(channel_dim, generator=generator) if permute else torch.arange(channel_dim)
        expand3x3_perm = torch.randperm(channel_dim, generator=generator) if permute else torch.arange(channel_dim)
        squeeze_perm = torch.cat([expand1x1_perm, expand3x3_perm + channel_dim], dim=0)

        expand1x1_weight.data = expand1x1_weight.data[expand1x1_perm]
        expand1x1_bias.data = expand1x1_bias.data[expand1x1_perm]
        expand3x3_weight.data = expand3x3_weight.data[expand3x3_perm]
        expand3x3_bias.data = expand3x3_bias.data[expand3x3_perm]

        squeeze_weight.data = squeeze_weight.data[:, squeeze_perm]
        squeeze_bias.data = squeeze_bias.data

    # the outputs of squeeze 1.11 are fed into both expand 1x1 and expand 3x3
    pas_expand("1.11.squeeze", "1.11.expand1x1", "1.11.expand3x3")
    # expand 1x1 is cat'd with expand 3x3 and fed into 1.12.squeeze
    pas_squeeze("1.11.expand1x1", "1.11.expand3x3", "1.12.squeeze")
    # Again, the outputs of the squeeze layer are sent to both expand layers
    pas_expand("1.12.squeeze", "1.12.expand1x1", "1.12.expand3x3")
    # Again, the outputs of the two expand layers are cat'd together
    pas_squeeze("1.12.expand1x1", "1.12.expand3x3", "1.14")

    if scale_factor is not None:
        # We want no consistent scaling up or down, so we want symmetry in log space around E[X] = 1.
        # (log normal mean is exp(mu + sigma^2/2))
        sigma = scale_factor**2
        mu = -sigma / 2
        # torch lognormal doesn't seem to let you pass a generator???
        distribution = LogNormal(loc=mu, scale=sigma)

        scaler = distribution.sample()
        state_dict["1.11.squeeze.weight"].data *= scaler
        state_dict["1.11.squeeze.bias"].data *= scaler
        state_dict["1.11.expand1x1.weight"].data /= scaler
        state_dict["1.11.expand3x3.weight"].data /= scaler

        scaler = distribution.sample()
        state_dict["1.11.expand1x1.weight"].data *= scaler
        state_dict["1.11.expand1x1.bias"].data *= scaler
        state_dict["1.11.expand3x3.weight"].data *= scaler
        state_dict["1.11.expand3x3.bias"].data *= scaler
        state_dict["1.12.squeeze.weight"].data /= scaler

        scaler = distribution.sample()
        state_dict["1.12.squeeze.weight"].data *= scaler
        state_dict["1.12.squeeze.bias"].data *= scaler
        state_dict["1.12.expand1x1.weight"].data /= scaler
        state_dict["1.12.expand3x3.weight"].data /= scaler

        scaler = distribution.sample()
        state_dict["1.12.expand1x1.weight"].data *= scaler
        state_dict["1.12.expand1x1.bias"].data *= scaler
        state_dict["1.12.expand3x3.weight"].data *= scaler
        state_dict["1.12.expand3x3.bias"].data *= scaler
        state_dict["1.14.weight"].data /= scaler


def scramble(state_dict, seed, scramble_count=1):
    scrambled_modules = []
    module_names = [
        "1.11.squeeze",
        "1.11.expand1x1",
        "1.11.expand3x3",
        "1.12.squeeze",
        "1.12.expand1x1",
        "1.12.expand3x3",
        "1.14",
    ]
    generator = torch.Generator().manual_seed(seed)
    while len(scrambled_modules) < scramble_count:
        module = random.choice(module_names)
        module_weight = state_dict[module + ".weight"]
        module_bias = state_dict.get(module + ".bias")
        if random.random() < 0.5 and (module, 0) not in scrambled_modules:
            perm = torch.randperm(module_weight.shape[0], generator=generator)
            module_weight.data = module_weight.data[perm]
            if module_bias is not None:
                module_bias.data = module_bias.data[perm]
            scrambled_modules.append((module, 0))
        elif (module, 0) not in scrambled_modules:
            perm = torch.randperm(module_weight.shape[1], generator=generator)
            module_weight.data = module_weight.data[:, perm]
            if module_bias is not None:
                module_bias.data = module_bias.data[:, perm]
            scrambled_modules.append((module, 1))

    generator = torch.Generator().manual_seed(seed)
    total_params = sum(p.numel() for name, p in state_dict.items() if any(m in name for m in module_names))
    params_to_scramble = int(total_params * scramble_count / 100)

    params_scrambled = 0
    while params_scrambled < params_to_scramble:
        module = random.choice(module_names)
        weight = state_dict[module + ".weight"]
        bias = state_dict.get(module + ".bias")

        # Calculate how many parameters left to scramble
        remaining = params_to_scramble - params_scrambled

        # Scramble weights
        weight_size = weight.numel()
        num_to_scramble = min(remaining, weight_size)
        if num_to_scramble > 0:
            # Get mean and std of weights
            mean = weight.mean()
            std = weight.std()
            # Generate random indices
            indices = torch.randperm(weight_size, generator=generator)[:num_to_scramble]
            # Replace with random values from same distribution
            weight.data.view(-1)[indices] = torch.normal(mean, std, size=(num_to_scramble,), generator=generator)
            params_scrambled += num_to_scramble
            remaining -= num_to_scramble

        # Scramble bias if it exists and we still have parameters to scramble
        if bias is not None and remaining > 0:
            bias_size = bias.numel()
            num_to_scramble = min(remaining, bias_size)
            if num_to_scramble > 0:
                mean = bias.mean()
                std = bias.std()
                indices = torch.randperm(bias_size, generator=generator)[:num_to_scramble]
                bias.data.view(-1)[indices] = torch.normal(mean, std, size=(num_to_scramble,), generator=generator)
                params_scrambled += num_to_scramble


def bad_permute_and_scale(state_dict, seed, permute=True, scale_factor=None):
    permutation = torch.randperm(512) if permute else torch.arange(512)
    scale_factor = torch.randn(1).abs().item() * scale_factor if scale_factor else 1

    state_dict["1.12.squeeze.weight"].data = state_dict["1.12.squeeze.weight"].data[:, permutation] * scale_factor


class ModelParamDataset(Dataset):
    def __init__(
        self,
        pairs,
        augmentation_factor,
        form="flattened",
        dtype=torch.bfloat16,
        scramble=False,
        scramble_count=0,
        **kwargs,
    ):
        assert not (scramble and augmentation_factor > 1), "Cannot scramble and augment"

        super().__init__()

        self.pairs = pairs
        self.augmentation_factor = augmentation_factor
        self.form = form
        self.dtype = dtype
        self.scramble = scramble
        self.scramble_count = scramble_count

    def __len__(self):
        return len(self.pairs) * self.augmentation_factor

    def __getitem__(self, idx):
        model_file, model_classes = self.pairs[idx // self.augmentation_factor]
        state_dict = torch.load(model_file, map_location=torch.device("cpu"), weights_only=True)

        if self.scramble:
            scramble(state_dict, seed=idx, scramble_count=self.scramble_count)

        if self.augmentation_factor > 1:
            permute_and_scale(state_dict, seed=idx)

        processed_state_dict = {}
        for k, v in state_dict.items():
            processed_k = k.replace(".", "_")

            processed_v = v.to(self.dtype).detach()
            if self.form == "flattened":
                processed_v = processed_v.flatten()

            processed_state_dict[processed_k] = processed_v

        labels = torch.zeros(1000)
        labels[model_classes] = 1

        return processed_state_dict, labels


def generate_datasets(
    model_dir,
    sizes,
    augmentation_factor,
    split_layer=10,
    accuracy_cutoff=0.9,
    verbose=True,
    test_ood=None,
    include_checkpoints=False,
    scramble_count=0,
    **kwargs,
):
    assert math.isclose(sum(sizes), 1), "Sizes must sum to 1"

    # todo: fold into __init__
    class_index = []
    with open("data/imagenet/LOC_synset_mapping.txt") as f:
        for line in f.readlines():
            class_name, class_description = line.strip().split(" ", 1)
            class_index.append(class_name)

    model_ids = set(os.path.splitext(os.path.basename(fname))[0] for fname in os.listdir(model_dir))

    id_model_files, two_class_model_files = [], []
    id_model_classes, two_class_model_classes = [], []
    for model_id in model_ids:
        if model_id.startswith("."):
            continue
        with open(os.path.join(model_dir, model_id) + ".json") as f:
            metadata = json.load(f)

        version = metadata["version"]

        is_checkpoint = (
            (version >= 2)  # no checkpointing until v2
            and (not metadata["early_stopped"])  # early stopping means this was the final model
            and (int(metadata["epochs_trained"]) < int(metadata["max_epochs"]))
        )  # as does training to max epochs

        try:
            if (
                (float(metadata["test_acc"]) < accuracy_cutoff)
                or (is_checkpoint and not include_checkpoints)
                or (split_layer != int(metadata["split_layer"]))
            ):
                continue
        except KeyError:
            print(metadata, model_id)
            raise

        model_file = os.path.join(model_dir, model_id) + ".pth"
        classes = [class_index.index(in_class) for in_class in metadata["target_classes"]]

        (id_model_files if len(classes) == 1 else two_class_model_files).append(model_file)
        (id_model_classes if len(classes) == 1 else two_class_model_classes).append(classes)

    train_size = round(sizes[0] * len(id_model_files))
    if test_ood == "twoclass":
        # if testing for two-class generalization, combine "val" and "test" into val
        val_size = len(id_model_files) - train_size
        test_size = 0
    else:
        val_size = round(sizes[1] * len(id_model_files))
        test_size = len(id_model_files) - train_size - val_size

    train, val, test = torch.utils.data.random_split(range(len(id_model_files)), [train_size, val_size, test_size])

    train_dataset = ModelParamDataset(
        [(id_model_files[idx], id_model_classes[idx]) for idx in train], augmentation_factor, **kwargs
    )
    val_dataset = ModelParamDataset([(id_model_files[idx], id_model_classes[idx]) for idx in val], 1, **kwargs)
    if test_ood == "twoclass":
        print(f"{len(two_class_model_files)} {test_ood} models")
        test_dataset = ModelParamDataset(
            [(two_class_model_files[idx], two_class_model_classes[idx]) for idx in range(len(two_class_model_files))],
            augmentation_factor=1,
            **kwargs,
        )
    elif test_ood == "scrambled":
        test_dataset = ModelParamDataset(
            [(id_model_files[idx], id_model_classes[idx]) for idx in test],
            augmentation_factor=1,
            scramble=True,
            scramble_count=scramble_count,
            **kwargs,
        )
    else:
        test_dataset = ModelParamDataset(
            [(id_model_files[idx], id_model_classes[idx]) for idx in test], augmentation_factor=1, **kwargs
        )

    if verbose:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(model_dir, batch_size=32, train_workers=4, test_workers=1, pin_memory=False, *args, **kwargs):
    train_dataset, val_dataset, test_dataset = generate_datasets(model_dir, *args, **kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=test_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=test_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float64  # errors compound otherwise
    data_dir = Path("/home/localdia/model_talk/imagenet_classifiers/data/")

    imagenet_dir = data_dir / "imagenet"
    imgn_datasets_dict = {
        x: datasets.ImageFolder(os.path.join(imagenet_dir, x), ft.DATA_TRANSFORMS[x]) for x in ["train", "val"]
    }

    test_loader, _, _ = get_dataloaders(
        "data/squeezenet_binary_classifiers",
        batch_size=32,
        sizes=[0.8, 0.1, 0.1],
        include_checkpoints=False,
        augmentation_factor=1,
        form=None,
        verbose=False,
    )
    inputs, labels = next(iter(test_loader))
    inputs = {k: v.to(device).to(dtype) for k, v in inputs.items()}
    labels = labels.to(device)

    print("Testing scaling implementation")
    assert test_augmentation(
        partial(permute_and_scale, permute=False, scale_factor=None), inputs, labels, imgn_datasets_dict
    ), "Bad test implementation"
    print("No-op test passed")

    assert test_augmentation(
        partial(permute_and_scale, permute=False, scale_factor=10), inputs, labels, imgn_datasets_dict
    ), "Bad scaling implementation"
    print("Scaling test passed")

    assert not test_augmentation(
        partial(bad_permute_and_scale, permute=False, scale_factor=10), inputs, labels, imgn_datasets_dict
    ), "Bad scaling test"
    print("Bad scaling test passed")

    print("Testing permutation implementation")

    assert test_augmentation(
        partial(permute_and_scale, permute=True, scale_factor=None), inputs, labels, imgn_datasets_dict
    ), "Bad permutation implementation"
    print("Permutation test passed")

    assert not test_augmentation(
        partial(bad_permute_and_scale, permute=True, scale_factor=None), inputs, labels, imgn_datasets_dict
    ), "Bad permutation test"
    print("Bad permutation test passed")

    print("Testing scaling+permutation implementation")
    assert test_augmentation(
        partial(permute_and_scale, permute=True, scale_factor=7), inputs, labels, imgn_datasets_dict
    ), "Bad scaling+permutation implementation"
    print("Scaling+permutation test passed")

    assert not test_augmentation(
        partial(bad_permute_and_scale, permute=True, scale_factor=7), inputs, labels, imgn_datasets_dict
    ), "Bad scaling+permutation test"
    print("Bad scaling+permutation test passed")
