import argparse
import itertools
import json
import os
import random
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime

import exrex
import numpy as np
import torch
import torch.nn as nn
from exrex import CATEGORIES
from jaxtyping import Bool, Float, Int
from torch import Tensor as T
from torch.nn.utils.rnn import pad_sequence

# Hacky way to force exrex to only use the ab vocab. Not 100% sure this works in all cases,
# so need to verify after generation
CATEGORIES["category_any"] = list("ab")


class RegexFamily:
    def __init__(self, vocab, regex_options):
        self.full_vocab = ["<pad>", "<bos>", "<eos>"] + vocab
        self.vocab = vocab
        self.regex_options = regex_options

    def tokenize(self, s: list[str]):
        return torch.tensor([self.full_vocab.index(c) for c in ["<bos>"] + s + ["<eos>"]], dtype=torch.long)

    def detokenize(self, tokens):
        return "".join(self.full_vocab[t] for t in tokens)


class RegexDataset(torch.utils.data.IterableDataset):
    def __init__(self, regex_family, seed=None):
        self.regex_family = regex_family
        self.seed = random.randint(0, 2**30) if seed is None else seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 1 if worker_info is None else worker_info.id
        return RegexDatasetIterator(self.seed * worker_id, self.regex_family)


class RegexDatasetIterator:
    def __init__(self, seed, regex_family):
        self.generator = random.Random(seed)
        self.regex_family = regex_family

    def __next__(self):
        if self.regex_family.regex_options:
            return self.generator.choice(self.regex_family.regex_options)
        raise NotImplementedError("No regex options provided")


def collate_fn(batch):
    str_inputs, inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return str_inputs, inputs_padded, labels


class RegexInputDataset(torch.utils.data.IterableDataset):
    def __init__(self, regex_family, regex, max_input_length, balanced_pct=0.2, seed=None):
        self.regex_family = regex_family
        self.regex_str = regex
        self.regex = re.compile(regex)
        self.generator = random.Random(random.randint(0, 2**30) if seed is None else seed)
        self.max_input_length = max_input_length
        self.balanced_pct = balanced_pct

    def __iter__(self):
        return self

    def generate_random(self):
        input_length = self.generator.randint(0, self.max_input_length)
        str_input = "".join(self.generator.choices(self.regex_family.vocab, k=input_length))
        return str_input

    def generate_match(self):
        while True:
            limit = self.generator.randint(1, self.max_input_length)
            candidate = exrex.getone(self.regex_str, limit)
            if self.regex.fullmatch(candidate):
                return candidate

    def generate_balanced(self):
        should_match = self.generator.random() < 0.5
        does_match = None

        while should_match != does_match:
            candidate = self.generate_match() if should_match else self.generate_random()
            does_match = (set(candidate) <= set(self.regex_family.vocab)) and bool(self.regex.fullmatch(candidate))

        return candidate

    def __next__(self):
        # Balanced not possible for match-nothing or match-anything
        if self.regex_str == r"(?!)" or self.regex_str == r".*" or self.regex_str == r"[ab]*":
            regex_input = self.generate_random()
        else:
            is_balanced = self.generator.random() < self.balanced_pct
            regex_input = self.generate_balanced() if is_balanced else self.generate_random()

        tokenized_input = self.regex_family.tokenize(list(regex_input))
        label = torch.tensor(int(bool(self.regex.fullmatch(regex_input))), dtype=torch.bfloat16)

        return regex_input, tokenized_input, label


class RegexGRU(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, num_layers=10):
        super(RegexGRU, self).__init__()
        # +2 to include <pad> and <bos> token
        self.embedding = nn.Embedding(input_vocab_size + 3, hidden_size, dtype=torch.bfloat16)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dtype=torch.bfloat16,
        )
        self.fc = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Int[T, "batch seq"]):
        eos_mask: Bool[T, "batch"] = x == 2
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        final_outs: Float[T, "batch d_model"] = rnn_out[eos_mask]
        output = self.sigmoid(self.fc(final_outs).squeeze(1))
        return output


def validate(model, validation_loader):
    max_samples = 10_000 // validation_loader.batch_size
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, (str_inputs, inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += (outputs > 0.5).eq(labels).sum().item()
            total_samples += inputs.size(0)
            if i > max_samples:
                break
    return total_loss / i, total_acc / total_samples


def train_model(training_args, regex_family, regex):
    train_dataset = RegexInputDataset(
        regex_family=regex_family,
        regex=regex,
        max_input_length=training_args.max_input_length,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_args.batch_size, collate_fn=collate_fn
    )

    validation_dataset = RegexInputDataset(
        regex_family=regex_family,
        regex=regex,
        max_input_length=training_args.max_input_length + 5,
    )
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=2048, collate_fn=collate_fn)

    model = RegexGRU(
        hidden_size=training_args.d_model, input_vocab_size=len(regex_family.vocab), num_layers=training_args.n_layers
    ).to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=training_args.lr, total_steps=training_args.total_steps
    )

    check_at = 10

    for step, (str_inputs, inputs, labels) in enumerate(train_loader):
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if loss.item() < training_args.train_target_loss and step > check_at:
            check_at = step + round(6_000 / training_args.batch_size)

            dt = datetime.now().strftime("%H:%M:%S")
            val_loss, val_acc = validate(model, validation_loader)

            if val_acc >= training_args.val_target_acc and loss.item() <= training_args.train_target_loss:
                print(f"{dt} {step=} loss={loss:.7f} val_loss={val_loss:.7f} val_acc={val_acc:.7f} regex={regex}")
                return model, step

        if step == training_args.total_steps:
            print(f"{dt} GIVING UP loss={loss:.7f} val_loss={val_loss:.7f} val_acc={val_acc:.7f} regex={regex} ")
            return None, step


@dataclass(frozen=True)
class TrainingArgs:
    batch_size: int
    max_input_length: int
    train_target_loss: float
    val_target_acc: float
    d_model: int
    n_layers: int
    lr: float
    total_steps: int
    seed: int
    verbose: bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_to_train", type=int, default=100)
    args = parser.parse_args()

    with open("regex_info.json", "r") as f:
        regex_info = json.load(f)

    batch_size = 1024
    training_args = TrainingArgs(
        batch_size=batch_size,
        max_input_length=15,
        train_target_loss=2e-4,
        val_target_acc=1,
        d_model=64,
        seed=random.randint(0, 2**28),
        n_layers=1,
        lr=0.01,
        total_steps=round(32 * 15_000 / batch_size),
        verbose=True,
    )

    regex_family = RegexFamily(
        vocab=["a", "b"], regex_options=list(r for r, info in regex_info.items() if info["states"] <= 3)
    )
    regex_dataset = RegexDataset(regex_family, seed=training_args.seed)

    model = RegexGRU(
        hidden_size=training_args.d_model, input_vocab_size=len(regex_family.vocab), num_layers=training_args.n_layers
    )
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    for i, regex in enumerate(regex_dataset):
        model_id = uuid.uuid4()

        model, steps = train_model(training_args, regex_family, regex)

        model_desc = {
            "regex": regex,
            "vocab": regex_family.vocab,
            "hyperparams": str(training_args),
            "model": str(model),
            "version": 1,
            "git_hash": "5a56d17",
        }
        if model:
            with open(f"data/{model_id}.json", "w") as f:
                json.dump(model_desc, f)
            torch.save(model.state_dict(), f"data/{model_id}.pth")
        if i == args.models_to_train:
            print(f"Trained {i} models, stopping.")
            break
