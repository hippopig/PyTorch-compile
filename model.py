"""A deliberately small Transformer used by all compiler experiments."""

from __future__ import annotations

import torch
from torch import nn


class TinyTransformerLM(nn.Module):
    """A decoder-style language model small enough to compile on a laptop GPU."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")

        sequence_length = token_ids.shape[1]
        # This assertion depends only on tensor metadata, so Dynamo can guard it.
        if sequence_length > self.max_seq_len:
            raise ValueError(f"sequence length must be <= {self.max_seq_len}")

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)

        # True entries are masked. Building this mask with PyTorch operations keeps
        # the work visible to the compiler and places it on the selected device.
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=token_ids.device,
        ).triu(diagonal=1)
        hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        return self.lm_head(self.final_norm(hidden))


def make_example_inputs(
    device: torch.device,
    *,
    batch_size: int = 4,
    sequence_length: int = 32,
    vocab_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic token IDs and next-token targets."""

    generator = torch.Generator(device=device).manual_seed(2026)
    tokens = torch.randint(
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=device,
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    return tokens, targets

