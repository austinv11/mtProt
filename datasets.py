from enum import Enum

import pandas as pd
import pytorch_lightning as pl


class UkBioBankDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.filepath = "data/BioBank.xlsx"

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            df = pd.read_excel(self.filepath, engine="openpyxl",
                               sheet_name="Training Set")

        if stage == "test":
            ...

