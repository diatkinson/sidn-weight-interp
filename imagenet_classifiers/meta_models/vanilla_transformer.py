import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN
        attn_output = self.self_attn(self.norm1(x))
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x


class TransformerModel(nn.Module):
    def __init__(self, d_model=768, num_layers=24, num_heads=12, d_ff=3072, dropout=0.1, max_len=8):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerClassifier(nn.Module):
    """
    Take each weight matrix + bias, flatten, and transform into d_model dim vector
    """

    def __init__(self, d_model=256, num_layers=10, num_heads=8, torch_dtype=torch.bfloat16, **kwargs):
        super().__init__()

        self.transformer = TransformerModel(
            d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=4 * d_model, dropout=0.0
        )

        input_model_dims = [
            ("1.11.squeeze", torch.Size([64, 384, 1, 1])),
            ("1.11.expand1x1", torch.Size([256, 64, 1, 1])),
            ("1.11.expand3x3", torch.Size([256, 64, 3, 3])),
            ("1.12.squeeze", torch.Size([64, 512, 1, 1])),
            ("1.12.expand1x1", torch.Size([256, 64, 1, 1])),
            ("1.12.expand3x3", torch.Size([256, 64, 3, 3])),
            ("1.14", torch.Size([2, 512, 1, 1])),
        ]
        self.layer_keys = [layer_key.replace(".", "_") for layer_key, _ in input_model_dims]
        feature_layers = {}
        for layer_key, sz in input_model_dims:
            feature_layers[layer_key.replace(".", "_")] = nn.Linear(np.prod(sz) + sz[0], d_model)
        self.features = nn.ModuleDict(feature_layers)
        self.classifier = nn.Linear(d_model * len(self.layer_keys), 1000)
        self.to(torch_dtype)

    def forward(self, mesa_state_dict, attention_mask=None, **kwargs):
        embeds = []
        for layer_key in self.layer_keys:
            layer = self.features[layer_key]
            weight_and_bias = torch.cat(
                [
                    tens.reshape(tens.shape[0], -1)
                    for layer_name, tens in mesa_state_dict.items()
                    if layer_key in layer_name
                ],
                dim=-1,
            )
            embed = layer(weight_and_bias)
            embeds.append(embed)

        embeds = torch.stack(embeds, dim=1)  # Concat along model dimension

        model_out = rearrange(self.transformer(embeds), "batch pos d_model -> batch (pos d_model)")

        return self.classifier(model_out)
