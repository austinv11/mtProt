import pytorch_lightning as pl


class MtEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = ...

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        ...
