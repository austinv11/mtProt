from enum import Enum

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader


class PandasDataset(Dataset):

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame):
        self.x = torch.from_numpy(x_df.values).detach().float()
        self.y = torch.from_numpy(y_df.values).detach().float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]


class UkBioBankDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.filepath = "data/BioBank.xlsx"

    def prepare_data(self):
        df = pd.read_excel(self.filepath, engine="openpyxl",
                           sheet_name="Metabolite Annotations")
        self.feature_names = df["BIOCHEMICAL"].values.tolist()
        self.num_features = len(self.feature_names)

    def setup(self, stage=None):
        if stage == "fit":
            df = pd.read_excel(self.filepath, engine="openpyxl",
                               sheet_name="Training Set")
            datapoints = df.shape[0]
            train_size = int(0.85 * datapoints)
            val_size = datapoints - train_size

            self.train_dataset, self.val_dataset = random_split(PandasDataset(df, df), [train_size, val_size])
        if stage == "test":
            df = pd.read_excel(self.filepath, engine="openpyxl",
                               sheet_name="Test Set")
            self.test_dataset = PandasDataset(df, df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128, shuffle=False)
