from torch import optim, nn
import lightning as L
from data import PLCODataModule
from lightning.pytorch.cli import LightningCLI
from model import CXRModel
import torch

torch.set_float32_matmul_precision('high')

# define the LightningModule
class LitModule(L.LightningModule):
    def __init__(self, lr, step_size, gamma):
        super().__init__()
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.model = CXRModel()

    def training_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"val_loss": loss, "val_mae": torch.mean(torch.abs(y_hat - y))}, on_epoch=True, on_step=False, sync_dist=True)

    # def on_test_start(self):
    #     self.results = 

    def test_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"test_loss": loss, "test_mae": torch.mean(torch.abs(y_hat - y))}, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.step_size, self.gamma)
        return [optimizer], [scheduler]

def cli_main():
    cli = LightningCLI(LitModule, PLCODataModule)

if __name__ == "__main__":
    cli_main()