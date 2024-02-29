from torch import optim, nn
import lightning as L
from data import PLCODataModule
from lightning.pytorch.cli import LightningCLI
from model import CXRModel
import torch
import pandas as pd

torch.set_float32_matmul_precision('high')

# define the LightningModule
class LitModule(L.LightningModule):
    def __init__(self, lr, step_size, gamma, task):
        super().__init__()
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.task = task
        if task == "classification":
            self.loss = nn.BCEWithLogitsLoss()
        elif task == "regression":
            self.loss = nn.Sequential([nn.Softplus(), nn.MSELoss()])
        else:
            raise ValueError("Inappropriate task")
        self.model = CXRModel()
        self.th = 0.5

    def training_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        loss = self.loss(y_hat, y)
        if self.task == "classification":
            self.log_dict({"val_loss": loss, "val_accuracy": ((y_hat > self.th) == y.bool()).float().mean()}, on_epoch=True, on_step=False, sync_dist=True)
        if self.task == "regression":
            self.log_dict({"val_loss": loss, "val_mae": torch.mean(torch.abs(y_hat - y))}, on_epoch=True, on_step=False, sync_dist=True)


    def on_test_start(self):
        self.results = pd.read_csv("data/label/test.csv")

    def test_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        loss = self.loss(y_hat, y)
        self.log_dict({"test_loss": loss, "test_mae": torch.mean(torch.abs(y_hat - y))}, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.step_size, self.gamma)
        return [optimizer], [scheduler]

def cli_main():
    cli = LightningCLI(LitModule, PLCODataModule)

if __name__ == "__main__":
    cli_main()