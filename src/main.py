from torch import optim, nn
import lightning as L
from data import PLCODataModule
from lightning.pytorch.cli import LightningCLI
from model import CXRModel
import torch
import pandas as pd
from sklearn.metrics import auc, accuracy_score, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

# define the LightningModule
class LitModule(L.LightningModule):
    def __init__(self, lr, step_size, gamma):
        super().__init__()
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.loss = nn.BCELoss()
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
        self.log_dict({"val_loss": loss, "val_accuracy": ((y_hat > self.th) == y.bool()).float().mean()}, on_epoch=True, on_step=False, sync_dist=True)

    def on_test_start(self):
        self.results = pd.read_csv("data/label/test.csv")
        self.true = []
        self.pred = []

    def test_step(self, batch, batch_idx):
        (tab, img), y = batch
        y_hat = self.model(tab, img)
        self.true.append(y.flatten().cpu().numpy())
        self.pred.append(y_hat.flatten().cpu().numpy())

    def on_test_end(self):
        self.true = np.concatenate(self.true)
        self.pred = np.concatenate(self.pred)

        # PRC and ROC curve
        PrecisionRecallDisplay.from_predictions(self.true, self.pred, plot_chance_level=True)
        plt.savefig("results/PRC.png")
        RocCurveDisplay.from_predictions(self.true, self.pred, plot_chance_level=True)
        plt.savefig("results/Roc_curve.png")

        # Best threshold
        precision, recall, ths = precision_recall_curve(self.true, self.pred)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1_score = np.max(f1_scores)
        th = ths[np.where(f1_scores==max_f1_score)[0][0]]

        # Confusion matrix
        self.pred = self.pred > th
        ConfusionMatrixDisplay.from_predictions(self.true, self.pred, cmap=plt.cm.Blues).ax_.set_title(f"Acc = {np.mean(self.true == self.pred) * 100:.2f}%")
        plt.savefig("results/confusion_matrix.png")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.step_size, self.gamma)
        return [optimizer], [scheduler]

def cli_main():
    cli = LightningCLI(LitModule, PLCODataModule)

if __name__ == "__main__":
    cli_main()