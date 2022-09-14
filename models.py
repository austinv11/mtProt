import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from torch import nn
import torch.functional as F


class MtEncoder(pl.LightningModule):
    def __init__(self,
                 num_features: int,
                 feature_names: list[str],
                 lr: float = 1e-3,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 num_layers: int = 3,
                 max_layer_size: int = 256,
                 latent_size: int = 32,
                 activation: str = "relu",
                 optimizer: str = "adam",
                 encoder_type: str = "vanilla",
                 ):
        super().__init__()

        self.activation = activation

        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation == "gelu":
            activation = nn.GELU()
        elif activation == "selu":
            activation = nn.SELU()

        if num_layers == 1:
            self.encoder = nn.Sequential(
                self._init_weights(nn.Linear(num_features, max_layer_size)),
                activation,
                self._init_weights(nn.Linear(max_layer_size, latent_size))
            )

            self.decoder = nn.Sequential(
                self._init_weights(nn.Linear(latent_size, max_layer_size)),
                activation,
                self._init_weights(nn.Linear(max_layer_size, num_features))
            )

        elif num_layers == 2:
            self.encoder = nn.Sequential(
                self._init_weights(nn.Linear(num_features, max_layer_size)),
                activation,
                self._init_weights(nn.Linear(max_layer_size, max_layer_size // 2)),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 2, latent_size))
            )

            self.decoder = nn.Sequential(
                self._init_weights(nn.Linear(latent_size, max_layer_size // 2)),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size)),
                activation,
                self._init_weights(nn.Linear(max_layer_size, num_features))
            )

        elif num_layers == 3:
            self.encoder = nn.Sequential(
                self._init_weights(nn.Linear(num_features, max_layer_size)),
                activation,
                self._init_weights(nn.Linear(max_layer_size, max_layer_size // 2)),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size // 4)),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 4, latent_size))
            )

            self.decoder = nn.Sequential(
                self._init_weights(nn.Linear(latent_size, max_layer_size // 4)),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 4, max_layer_size // 2)),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size)),
                activation,
                self._init_weights(nn.Linear(max_layer_size, num_features))
            )

        else:
            raise ValueError("num_layers must be 1, 2, or 3")

        # Metrics
        self.r2_score = torchmetrics.R2Score(num_outputs=num_features,
                                             multioutput="raw_values")
        self.feature_names = feature_names
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.encoder_type = encoder_type

        self.save_hyperparameters()

    def _init_weights(self, layer):
        if self.activation != "selu":
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(self.activation))

        return layer

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.forward_encoder(x))

    def _kl_divergence(self, rho_pred, rho=0.05):
        # Rho is the sparsity parameter
        rho_hat = torch.mean(F.sigmoid(rho_pred), dim=1)  # Convert to probabilities
        rho = torch.ones_like(rho_hat) * rho
        return torch.sum(
            rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )

    def _sparse_loss(self, values, rho=0.05):
        # Rho is the sparsity parameter
        loss = 0
        for child in self.children():
            values = child(values)
            loss += self._kl_divergence(values, rho=rho)
        return loss

    def _contractive_loss(self, mse, x, xhat, lambda_=1e-4, reduction=torch.sum):
        # https://github.com/AlexPasqua/Autoencoders/blob/main/src/custom_losses.py
        # https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
        # Lambda is the regularization parameter
        # reduction can be torch.mean or torch.sum
        encoder_units = reversed([child for child in self.children() if isinstance(child, nn.Linear)])
        encoder_weights = [child.weight for child in encoder_units]
        contractive_loss = torch.mean(mse + (lambda_ * torch.norm(torch.chain_matmul(*encoder_weights))), 0)
        return reduction(contractive_loss)

    def process_batch(self, batch, is_training=False):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        if self.encoder_type == "sparse" and is_training:
            # https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/
            sparsity = self._sparse_loss(x)
            beta = 0.001  # Weight for the sparsity penalty
            loss += beta * sparsity
        elif self.encoder_type == "contractive" and is_training:
            loss = self._contractive_loss(loss, x, y_hat)

        r2 = self.r2_score(y_hat, y)
        r2_table = wandb.Table(data=[[feat, r2_val] for (feat, r2_val) in zip(self.feature_names, r2)],
                               columns=["feature", "r2"])
        return loss, r2_table

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
           wandb.define_metric('train_r2', summary='last', goal="maximize")

        loss, r2 = self.process_batch(batch, is_training=True)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_r2",
                 wandb.plot.bar(
                     r2, "feature", "r2",
                     title="R2 Score by Feature"
                 ),
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
           wandb.define_metric('val_r2', summary='last', goal="maximize")

        loss, r2 = self.process_batch(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_r2",
                 wandb.plot.bar(
                     r2, "feature", "r2",
                     title="R2 Score by Feature"
                 ),
                 on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
           wandb.define_metric('test_r2', summary='last', goal="maximize")

        loss, r2 = self.process_batch(batch)

        self.log("test_loss", loss)
        self.log("test_r2",
                 wandb.plot.bar(
                     r2, "feature", "r2",
                     title="R2 Score by Feature"
                 ))

        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.momentum > 0)
        elif self.optimizer == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        elif self.optimizer == "adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "radam":
            return torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
