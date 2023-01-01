import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class PandasDataset(Dataset):

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame):
        self.x = torch.from_numpy(x_df.values).detach().float()
        self.y = torch.from_numpy(y_df.values).detach().float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]


# TODO: Multitask DataModule (randomly split columns to tasks)
# TODO: Check preprocessing from VAE paper
# TODO: Can we weigh data based on when it was collected?

class UkBioBankDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 128, only_nonderived: bool = True, pseudo_targets: bool = False):
        super().__init__()
        self.filepath = "data/BioBank.xlsx"
        self.meta_filepath = "data/264079_file03.xlsx"  # Supplemental Table S1: https://www.medrxiv.org/content/10.1101/2021.09.24.21264079v2.supplementary-material
        self.batch_size = batch_size
        self.only_nonderived = only_nonderived
        self.pseudo_targets = pseudo_targets  # Split input into fake labels

    def prepare_data(self):
        df = pd.read_excel(self.filepath, engine="openpyxl",
                           sheet_name="Metabolite Annotations")
        nightengale_metadata = pd.read_excel(self.meta_filepath, engine="openpyxl",
                                             sheet_name="Table S1", skiprows=2)
        nightengale_metadata = nightengale_metadata.drop(columns=["Biomarker", "Units", "Group", "Sub-group", "UKB Field ID", "QC Flag Field ID"])
        nightengale_metadata = nightengale_metadata.rename(columns={"Description": "BIOCHEMICAL"})
        df = df.merge(nightengale_metadata, on="BIOCHEMICAL", how="left")

        self.is_nonderived = (df['Type'] == "Non-derived").values.tolist()
        if self.only_nonderived:
            self.feature_names = [name for name, is_nonderived in zip(df["BIOCHEMICAL"].values.tolist(), self.is_nonderived) if is_nonderived]
        else:
            self.feature_names = df["BIOCHEMICAL"].values.tolist()

        if self.pseudo_targets:
            # Randomly select half of features indices as targets
            self.target_mask = np.random.choice([False, True], size=len(self.feature_names), replace=True)
            self.feature_mask = np.logical_not(self.target_mask)
        else:
            self.target_mask = [False] * len(self.feature_names)  # TODO: Real targets
            self.feature_mask = [True] * len(self.feature_names)

        self.target_names = [name for name, is_target in zip(self.feature_names, self.target_mask) if is_target]
        if len(self.target_names) == 0:
            self.target_names = ["None"]
        self.feature_names = [name for name, is_feature in zip(self.feature_names, self.feature_mask) if is_feature]

        self.num_features = len(self.feature_names)
        self.num_targets = len(self.target_names)

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            df = pd.read_excel(self.filepath, engine="openpyxl",
                               sheet_name="Training Set")
            if self.only_nonderived:
                df = df[df.columns[self.is_nonderived]]
            data_df = df[df.columns[self.feature_mask]]
            target_df = df[df.columns[self.target_mask]]
            if target_df.shape[1] == 0:
                target_df = pd.DataFrame(np.zeros((data_df.shape[0], 1)))
            datapoints = df.shape[0]
            train_size = int(0.85 * datapoints)
            val_size = datapoints - train_size

            self.train_dataset, self.val_dataset = random_split(PandasDataset(data_df, target_df), [train_size, val_size])
        if stage == "test":
            df = pd.read_excel(self.filepath, engine="openpyxl",
                               sheet_name="Testing Set")
            if self.only_nonderived:
                df = df[df.columns[self.is_nonderived]]
            data_df = df[df.columns[self.feature_mask]]
            target_df = df[df.columns[self.target_mask]]
            if target_df.shape[1] == 0:
                target_df = pd.DataFrame(np.zeros((data_df.shape[0], 1)))
            self.test_dataset = PandasDataset(data_df, target_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
