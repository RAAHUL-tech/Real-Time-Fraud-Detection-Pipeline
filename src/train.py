from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.datamodule import FraudDataModule
from src.model import FraudModel
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Confusion matrix saved to {save_path}")

def test_model(model, datamodule):
    datamodule.setup()
    test_loader = datamodule.val_dataloader()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"]))
    plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    datamodule = FraudDataModule(data_dir="data/processed", batch_size=256)
    model = FraudModel(input_dim=29)

    logger = CSVLogger("logs", name="fraud_detection")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint",
        dirpath="checkpoints/"
    )

    early_stop_cb = EarlyStopping(
        monitor="val_auroc",
        patience=3,
        mode="max",
        verbose=True
    )

    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        log_every_n_steps=1
    )

    trainer.fit(model, datamodule)

    # Load best model for testing
    best_model = FraudModel.load_from_checkpoint(checkpoint_cb.best_model_path)
    test_model(best_model, datamodule)
