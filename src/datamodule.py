import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule

class FraudDataModule(LightningDataModule):
    def __init__(self, data_dir="data/processed", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load data
        X_train = pd.read_csv(f"{self.data_dir}/X_train.csv").values
        y_train = pd.read_csv(f"{self.data_dir}/y_train.csv").values.ravel()
        X_test = pd.read_csv(f"{self.data_dir}/X_test.csv").values
        y_test = pd.read_csv(f"{self.data_dir}/y_test.csv").values.ravel()

        # Convert to tensors
        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                           torch.tensor(y_train, dtype=torch.float32))
        self.test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                          torch.tensor(y_test, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
