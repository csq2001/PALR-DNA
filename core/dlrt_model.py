from __future__ import annotations

import math

import torch
import torch.nn as nn


class DNAByteNet(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        target_bytes: int | None = None,
        max_len: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.target_bytes = target_bytes or latent_channels
        self.base_embedding = nn.Embedding(5, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.channel_queries = nn.Parameter(torch.randn(self.target_bytes, d_model) / math.sqrt(d_model))
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
        )

    def forward(self, input_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.base_embedding(input_ids) + self.position_embedding[:, : input_ids.shape[1]]
        padding_mask = ~valid_mask
        memory = self.encoder(x, src_key_padding_mask=padding_mask)
        queries = self.channel_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        channel_features, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.head(self.norm(channel_features))

