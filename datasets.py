from typing import Tuple, List

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader, ChainDataset


class PandasDataset(Dataset):

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame):
        self.x = torch.from_numpy(x_df.values).detach().float()
        self.y = torch.from_numpy(y_df.values).detach().float()
        self.x_labels = x_df.columns.values.tolist()
        self.y_labels = y_df.columns.values.tolist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]

    def filter_columns(self, cols_to_keep: List[str]):
        self.x = self.x[:, [self.x_labels.index(col) for col in cols_to_keep if col in self.x_labels]]
        self.x_labels = [x for x in self.x_labels if x in cols_to_keep]

        self.y = self.y[:, [self.y_labels.index(col) for col in cols_to_keep if col in self.y_labels]]
        self.y_labels = [y for y in self.y_labels if y in cols_to_keep]


def load_summarized_experiment(feature_cols: List[str],
                               target_cols: List[str],
                               file: str,
                               experiment: str) -> Tuple[int, PandasDataset]:
    assay = pd.read_excel(file, engine="openpyxl", sheet_name="assay")
    assay = assay.set_index(assay.columns.values[0]).T
    # Assay now has rows representing individuals and columns representing metabolites

    # Metadata for each metabolite
    rowData = pd.read_excel(file, engine="openpyxl", sheet_name="rowData")
    rowData = rowData.set_index(rowData.columns.values[0])

    column_to_select = "Biomarker_name" if "Biomarker_name" in rowData.columns.values else "title"
    assay.columns = [c.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower() for c in rowData[column_to_select].values]

    # Filter out missing columns
    feature_cols = [col for col in feature_cols if col in assay.columns]
    target_cols = [col for col in target_cols if col in assay.columns]

    # Add indicator for experiment
    assay["experiment"] = experiment

    # Add zeros to allow for ignoring targets
    assay["zero"] = 0.0

    # Metadata for each individual
    #colData = pd.read_excel(file, engine="openpyxl", sheet_name="colData")
    #colData = colData.set_index(colData.columns.values[0])

    # Add subject identifiers
    assay["subject"] = [f"{experiment}_{i}" for i in assay.index.values]  #colData.index.values
    assay = assay.set_index("subject")

    return assay.shape[0], PandasDataset(assay[feature_cols], assay[target_cols])


def adni_data(feature_cols: List[str],
              target_cols: List[str],
              filepath: str = "data/ADNI_norm.xlsx") -> Tuple[int, PandasDataset]:
    return load_summarized_experiment(feature_cols, target_cols, filepath, "ADNI")


def tulsa_data(feature_cols: List[str],
               target_cols: List[str],
               filepath: str = "data/Tulsa_norm.xlsx") -> Tuple[int, PandasDataset]:
    return load_summarized_experiment(feature_cols, target_cols, filepath, "Tulsa")


def biobank_data(feature_cols: List[str],
                 target_cols: List[str],
                 filepath: str = "data/BioBank_norm.xlsx") -> Tuple[int, PandasDataset]:
    return load_summarized_experiment(feature_cols, target_cols, filepath, "UK_BioBank")


class NightingaleDataModule(pl.LightningDataModule):

    def __init__(self,
                 on_gpu: bool,
                 batch_size: int = 128,
                 train_proportion: float = 0.85,
                 val_proportion: float = 0.1,
                 pseudo_targets: bool = False,
                 meta_filepath: str = "data/264079_file03.xlsx",
                 generator_seed: int = 1234567890):
        super().__init__()
        self.on_gpu = on_gpu
        self.batch_size = batch_size
        self.train_proportion = train_proportion
        self.val_proportion = val_proportion
        self.pseudo_targets = pseudo_targets
        self.meta_filepath = meta_filepath
        self.generator = torch.Generator().manual_seed(generator_seed)

    def prepare_data(self):
        nightengale_metadata = pd.read_excel(self.meta_filepath, engine="openpyxl",
                                             sheet_name="Table S1", skiprows=2)
        nightengale_metadata = nightengale_metadata.drop(
            columns=["Units", "Group", "Sub-group", "UKB Field ID", "QC Flag Field ID"])
        nightengale_metadata = nightengale_metadata.rename(columns={"Description": "metabolite"})

        non_derived_metabolites = [m.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower() for m in nightengale_metadata[nightengale_metadata["Type"] == "Non-derived"]["metabolite"].values.tolist()]

        if self.pseudo_targets:
            # Randomly select metabolites to be targets
            targets = torch.randperm(len(non_derived_metabolites), generator=self.generator).tolist()[:len(non_derived_metabolites)//2]
            targets = [non_derived_metabolites[i] for i in targets]
            non_derived_metabolites = [m for m in non_derived_metabolites if m not in targets]
        else:
            # FIXME: This is a placeholder
            targets = ["zero"]

        adni_size, adni = adni_data(non_derived_metabolites, targets)
        tulsa_size, tulsa = tulsa_data(non_derived_metabolites, targets)
        biobank_size, biobank = biobank_data(non_derived_metabolites, targets)

        all_columns = (set(adni.x_labels) | set(adni.y_labels)) & (set(tulsa.x_labels) | set(tulsa.y_labels)) & (set(biobank.x_labels) | set(biobank.y_labels))
        adni.filter_columns(all_columns)
        tulsa.filter_columns(all_columns)
        biobank.filter_columns(all_columns)

        self.feature_names = biobank.x_labels
        self.target_names = biobank.y_labels

        adni_train_size = int(adni_size * self.train_proportion)
        adni_train, self.adni_test_data = random_split(adni, generator=self.generator,
                                             lengths=[adni_train_size, adni_size - adni_train_size])

        tulsa_train_size = int(tulsa_size * self.train_proportion)
        tulsa_train, self.tulsa_test_data = random_split(tulsa, generator=self.generator,
                                               lengths=[tulsa_train_size, tulsa_size - tulsa_train_size])

        biobank_train_size = int(biobank_size * self.train_proportion)
        biobank_train, self.biobank_test_data = random_split(biobank, generator=self.generator,
                                                   lengths=[biobank_train_size, biobank_size - biobank_train_size])

        adni_val_size = int(adni_train_size * self.val_proportion)
        adni_train_size = adni_train_size - adni_val_size
        self.adni_train_data, self.adni_val_data = random_split(adni_train, generator=self.generator,
                                            lengths=[adni_train_size, adni_val_size])

        tulsa_val_size = int(tulsa_train_size * self.val_proportion)
        tulsa_train_size = tulsa_train_size - tulsa_val_size
        self.tulsa_train_data, self.tulsa_val_data = random_split(tulsa_train, generator=self.generator,
                                              lengths=[tulsa_train_size, tulsa_val_size])

        biobank_val_size = int(biobank_train_size * self.val_proportion)
        biobank_train_size = biobank_train_size - biobank_val_size
        self.biobank_train_data, self.biobank_val_data = random_split(biobank_train, generator=self.generator,
                                                  lengths=[biobank_train_size, biobank_val_size])


    def setup(self, stage: str = None):
        if stage is None or stage == "fit":
            return

        if stage == "test":
            return

        raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        adni_train_loader = DataLoader(self.adni_train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=self.on_gpu,
                                       num_workers=2 if self.on_gpu else 0)
        tulsa_train_loader = DataLoader(self.tulsa_train_data,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        pin_memory=self.on_gpu,
                                        num_workers=2 if self.on_gpu else 0)
        biobank_train_loader = DataLoader(self.biobank_train_data,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          pin_memory=self.on_gpu,
                                          num_workers=2 if self.on_gpu else 0)

        # Defer to Pytorch-Lightning deal with this to deal with the different sizes
        return [adni_train_loader, tulsa_train_loader, biobank_train_loader]

    def val_dataloader(self):
        adni_val_loader = DataLoader(self.adni_val_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     pin_memory=self.on_gpu,
                                     num_workers=2 if self.on_gpu else 0)
        tulsa_val_loader = DataLoader(self.tulsa_val_data,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=self.on_gpu,
                                      num_workers=2 if self.on_gpu else 0)
        biobank_val_loader = DataLoader(self.biobank_val_data,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        pin_memory=self.on_gpu,
                                        num_workers=2 if self.on_gpu else 0)

        # Defer to Pytorch-Lightning deal with this to deal with the different sizes
        return [adni_val_loader, tulsa_val_loader, biobank_val_loader]

    def test_dataloader(self):
        adni_test_loader = DataLoader(self.adni_test_data,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=self.on_gpu,
                                      num_workers=2 if self.on_gpu else 0)
        tulsa_test_loader = DataLoader(self.tulsa_test_data,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=self.on_gpu,
                                       num_workers=2 if self.on_gpu else 0)
        biobank_test_loader = DataLoader(self.biobank_test_data,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         pin_memory=self.on_gpu,
                                         num_workers=2 if self.on_gpu else 0)

        # Consider every datapoint exactly once
        return ChainDataset([adni_test_loader, tulsa_test_loader, biobank_test_loader])


if __name__ == "__main__":
    dataset = NightingaleDataModule(
        on_gpu=False,
        pseudo_targets=False
    )
    dataset.prepare_data()
