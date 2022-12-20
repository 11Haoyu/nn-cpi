import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchdata import datapipes as dp
from sklearn.preprocessing import MinMaxScaler


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.rnn = nn.GRU(1, 8, 2, batch_first=True)
        self.proj = nn.Linear(8, 1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        x, _ = self.rnn(X)
        x = self.proj(x[:, -1, :])
        loss = F.mse_loss(x.view(-1), y.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        x, _ = self.rnn(X)
        x = self.proj(x[:, -1, :])
        loss = F.mse_loss(x.view(-1), y.view(-1))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    dataset_train = pd.read_csv("./data/NSE-TATAGLOBAL.csv")
    dataset_train = dataset_train.sort_values(by="Date").reset_index(drop=True)
    training_set = dataset_train.iloc[:, 1:2].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(60, training_set_scaled.shape[0]):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X, y = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    X_train = X[:1200]
    X_val = X[1200:]
    y_train = y[:1200]
    y_val = y[1200:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    pipe = dp.iter.IterableWrapper(zip(X_train, y_train))
    pipe = pipe.shuffle()

    train_dataloader = DataLoader(
        dataset=pipe,
        batch_size=8,
        num_workers=4,
        shuffle=True,
    )

    pipe = dp.iter.IterableWrapper(zip(X_val, y_val))

    val_dataloader = DataLoader(dataset=pipe, batch_size=8, num_workers=4)

    model = Model()

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
