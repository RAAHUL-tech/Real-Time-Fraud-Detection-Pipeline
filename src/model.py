import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall

class FraudModel(LightningModule):
    def __init__(self, input_dim=29, hidden_dim=64, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auroc", self.auroc(preds, y.int()), prog_bar=True)
        self.log("train_acc", self.accuracy(preds, y.int()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", self.auroc(preds, y.int()), prog_bar=True)
        self.log("val_acc", self.accuracy(preds, y.int()), prog_bar=True)
        self.log("val_precision", self.precision(preds, y.int()), prog_bar=True)
        self.log("val_recall", self.recall(preds, y.int()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
