import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class DeepCPIBase(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        add_noise: float = 0.,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.add_noise = add_noise
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx: int):
        x, y = batch

        if self.add_noise > 0:
            x = x + torch.randn_like(x) * self.add_noise

        x = self.forward(x)

        loss_mse = F.mse_loss(x.view(-1), y.view(-1))

        self.log("train/loss_mse", loss_mse)

        return loss_mse

    def validation_step(self, batch, batch_idx: int):
        x, y = batch

        x = self.forward(x)

        loss_mse = F.mse_loss(x.view(-1), y.view(-1))

        self.log("val/loss_mse", loss_mse)

        return loss_mse

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
