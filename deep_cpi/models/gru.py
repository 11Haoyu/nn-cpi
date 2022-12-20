import torch
from torch import nn

from .base import DeepCPIBase


class DeepCPIGRU(DeepCPIBase):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.,
        bidirectional: bool = False,
        add_noise: float = 0.,
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            in_channels=in_channels,
            add_noise=add_noise,
            learning_rate=learning_rate,
        )

        self.gru = nn.GRU(
            in_channels, hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.predictor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.predictor(x[:, -1, :])
        return x
