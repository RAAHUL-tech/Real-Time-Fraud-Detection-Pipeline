from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datamodule import FraudDataModule
from model import FraudModel
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.onnx
from dotenv import load_dotenv
import json
import hydra
from omegaconf import DictConfig
import wandb
from pytorch_lightning.loggers import WandbLogger
import mlflow
import mlflow.pytorch

def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
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

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"]))
    plot_confusion_matrix(y_true, y_pred)
    return precision, recall, f1


def export_to_onnx(model, save_path="models/fraud_model.onnx"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dummy_input = torch.randn(1, 29)  # input_dim = 29
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[INFO] Model exported to ONNX at: {save_path}")


def export_io_schema(save_path="models/fraud_model_schema.json"):
    schema = {
        "input": {
            "name": "input",
            "shape": ["batch_size", 29],
            "dtype": "float32",
            "description": "29 input features (V1–V28, Amount)"
        },
        "output": {
            "name": "output",
            "shape": ["batch_size", 1],
            "dtype": "float32",
            "description": "Sigmoid probability of fraud"
        }
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(schema, f, indent=4)
    print(f"[INFO] ONNX schema exported to: {save_path}")

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    load_dotenv()
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    seed_everything(cfg.train.seed)
    datamodule = FraudDataModule(data_dir=cfg.data.path, batch_size=cfg.train.batch_size)
    model = FraudModel(input_dim=cfg.data.input_dim, lr=cfg.train.lr)

    # Init W&B
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        log_model=cfg.wandb.log_model,
        job_type=cfg.wandb.job_type,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        name=f"dropout_{cfg.train.dropout}_lr_{cfg.train.lr}"
    )



    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        filename=cfg.checkpoint.filename,
        dirpath=cfg.checkpoint.dirpath
    )

    early_stop_cb = EarlyStopping(
        monitor=cfg.early_stop.monitor,
        patience=cfg.early_stop.patience,
        mode=cfg.early_stop.mode
    )

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        log_every_n_steps=1
    )

    trainer.fit(model, datamodule)

    # Load best model for testing
    best_model = FraudModel.load_from_checkpoint(checkpoint_cb.best_model_path)
    precision, recall, f1 = test_model(best_model, datamodule)
    export_to_onnx(best_model, save_path="models/fraud_model.onnx")
    export_io_schema(save_path="models/fraud_model_schema.json")
    
    # MLflow: model registry and deployment
    mlflow.set_tracking_uri("file:///D:/PG_PROJECTS/Real-Time-Fraud-Detection-Pipeline/mlruns")
    mlflow.set_experiment("fraud_detection_experiment")

    with mlflow.start_run(run_name=f"dropout_{cfg.train.dropout}_lr_{cfg.train.lr}"):
        mlflow.log_params({
            "dropout": cfg.train.dropout,
            "lr": cfg.train.lr,
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.max_epochs
        })
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Register model
        mlflow.pytorch.log_model(best_model, artifact_path="model", registered_model_name="FraudDetectionModel")

    wandb.finish()

if __name__ == "__main__":
    main()