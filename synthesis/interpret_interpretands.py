import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor as T
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from train_interpretands import RegexGRU

# TODO: save tokenizers


StateDict = dict[str, T]


class RegexDataset(Dataset):
    """
    {
        "embedding.weight": torch.Size([5, 16]),
        "rnn.weight_ih_l0": torch.Size([48, 16]),
        "rnn.weight_hh_l0": torch.Size([48, 16]),
        "rnn.bias_ih_l0": torch.Size([48]),
        "rnn.bias_hh_l0": torch.Size([48]),
        "rnn.weight_ih_l1": torch.Size([48, 16]),
        "rnn.weight_hh_l1": torch.Size([48, 16]),
        "rnn.bias_ih_l1": torch.Size([48]),
        "rnn.bias_hh_l1": torch.Size([48]),
        "rnn.weight_ih_l2": torch.Size([48, 16]),
        "rnn.weight_hh_l2": torch.Size([48, 16]),
        "rnn.bias_ih_l2": torch.Size([48]),
        "rnn.bias_hh_l2": torch.Size([48]),
        "fc.weight": torch.Size([1, 16]),
        "fc.bias": torch.Size([1]),
    }
    """

    def __init__(
        self,
        data_dir: str = "data",
        output_form: str = "unprocessed",
        filter_fn: Callable[[dict[str, Any]], bool] = lambda _: True,
    ):
        self.data_dir = data_dir
        self.output_form = output_form

        self.base_model_paths = []
        for f in os.listdir(data_dir):
            model_path = os.path.join(data_dir, f.split(".")[0]) + ".pth"
            if f.endswith(".json"):
                with open(os.path.join(data_dir, f), "r") as f:
                    metadatum = json.load(f)
                    if filter_fn(metadatum):
                        self.base_model_paths.append((model_path, metadatum["regex"]))

    def __len__(self):
        return len(self.base_model_paths)

    def __getitem__(self, idx: int) -> tuple[StateDict, str]:
        state_dict_path, regex = self.base_model_paths[idx]
        state_dict = torch.load(state_dict_path, weights_only=True)

        match self.output_form:
            case "unprocessed":
                pass
            case "flattened":
                state_dict = torch.cat([p.flatten() for p in state_dict.values()])
            case "tokenized":
                tokens = []
                for k, v in state_dict.items():
                    flattened_v = v.flatten()
                    pad_count = 48 - (flattened_v.shape[0] % 48)
                    if pad_count > 0:
                        flattened_v = F.pad(flattened_v, (0, pad_count))

                    tokens.append(flattened_v.reshape(-1, 48))
                state_dict = torch.cat(tokens, dim=0)
            case _:
                raise ValueError(f"Invalid output form: {self.output_form}")

        return state_dict, regex


class MLPInterpreter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: T) -> T:
        x = self.input_layer(x)
        x = self.gelu(x)
        for layer in self.layers:
            x = self.gelu(layer(x))
        x = self.output_layer(x)
        return self.sigmoid(x)


@dataclass(frozen=True)
class TransformerConfig:
    n_layers: int
    n_heads: int
    d_model: int
    dtype: torch.dtype


class TransformerInterpreter(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.embedder = nn.Linear(48, cfg.d_model, dtype=cfg.dtype)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            norm_first=True,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
            norm=nn.RMSNorm(cfg.d_model),
        ).to(cfg.dtype)
        self.output_proj = nn.Linear(cfg.d_model, 1, dtype=cfg.dtype)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Float[T, "batch seq 48"]) -> Float[T, "batch"]:
        x = self.embedder(x)
        x = self.transformer(x)
        x = self.output_proj(x[:, -1, :])
        return self.sigmoid(x)


def train_mlp_interpreter(model: MLPInterpreter, dataloader: DataLoader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(100):
        for step, (state_dict, regex) in enumerate(tqdm(dataloader, desc="Training", leave=True)):
            label = torch.tensor([[bool(r == "(aa)+")] for r in regex]).to("cuda").to(torch.bfloat16)
            output = model(state_dict.to("cuda"))
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                tqdm.write(f"Epoch {epoch}, Step {step}, loss: {loss.item()}")


if __name__ == "__main__":
    dataset = RegexDataset(output_form="tokenized")
    dataloader = DataLoader(dataset, batch_size=2_000)

    # interpreter = MLPInterpreter(4993, 2_000, 6).to("cuda").to(torch.bfloat16)
    transformer_cfg = TransformerConfig(n_layers=10, n_heads=8, d_model=512, dtype=torch.bfloat16)
    interpreter = TransformerInterpreter(transformer_cfg).to("cuda")
    print(f"params: {sum(p.numel() for p in interpreter.parameters()):.2e}")
    train_mlp_interpreter(interpreter, dataloader)
