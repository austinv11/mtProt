import math

import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from torch import nn
import torch.functional as F


# Implemenetation of the ConcreteAutoEncoder from the paper:
# https://arxiv.org/abs/1901.09346
# Based on author tensorflow implementation at: https://github.com/mfbalin/concrete-autoencoders
class ConcreteEncoder(nn.Module):

    def __init__(self,
                 input_size: int,
                 K: int,  # Number of features to select
                 alpha: float = 0.99999,  # Temperature schedule
                 start_temperature: float = 10.0,  # Starting temperature for the concrete encoder
                 end_temperature: float = 0.1,  # Ending temperature for the concrete encoder
                 ):
        super().__init__()

        self.input_size = input_size
        self.K = K
        self.start_temp = start_temperature
        self.end_temp = end_temperature
        self.alpha = alpha
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0, 1)

        self.temp = nn.Parameter(torch.tensor(start_temperature), requires_grad=False)
        self.logits = nn.Parameter(torch.zeros(K), requires_grad=True)
        self.selections = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.learn = nn.Parameter(torch.tensor(1), requires_grad=False)

        nn.init.xavier_normal_(self.logits)

    @classmethod
    def calculate_alpha(cls, num_epochs: int, features: int, batch_size: int, start_temp: float = 10.0, end_temp: float = 0.1):
        # Calculate ideal alpha for the concrete encoder
        # Ref: https://github.com/mfbalin/Concrete-Autoencoders/blob/master/concrete_autoencoder/concrete_autoencoder/__init__.py#L89
        return math.exp(math.log(end_temp / start_temp) / num_epochs * (features + batch_size - 1) // batch_size)

    def forward(self, x, training=False):
        self.temp.data = torch.clamp(self.temp.data * self.alpha, self.end_temp, self.start_temp)
        noisy_logits = (self.logits + self.gumbel_dist.sample(self.logits.shape)) / self.temp
        samples = F.softmax(noisy_logits, dim=0)
        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=0), self.logits.shape[0])

        if training:
            self.selections.data = samples
        else:
            self.selections.data = discrete_logits

        return torch.dot(x, torch.transpose(self.selections, 0, 1))


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
                 dropout: float = 0.0,
                 corruption_prob: float = 0.0
                 ):
        super().__init__()

        self.activation = activation

        if dropout == 0.0:
            dropout_cls = nn.Identity
        elif activation == "selu":
            dropout_cls = nn.AlphaDropout
        else:
            dropout_cls = nn.Dropout

        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation == "gelu":
            activation = nn.GELU()
        elif activation == "selu":
            activation = nn.SELU()

        if num_layers == 1:
            if encoder_type == 'concrete':
                self.encoder = ConcreteEncoder(
                    input_size=num_features,
                    K=latent_size,
                )
            else:
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation,
                    self._init_weights(nn.Linear(max_layer_size, latent_size))
                )

            self.decoder = nn.Sequential(
                self._init_weights(nn.Linear(latent_size, max_layer_size)),
                dropout_cls(dropout),
                activation,
                self._init_weights(nn.Linear(max_layer_size, num_features))
            )

        elif num_layers == 2:
            if encoder_type == 'concrete':
                self.encoder = ConcreteEncoder(
                    input_size=num_features,
                    K=latent_size,
                )
            else:
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation,
                    self._init_weights(nn.Linear(max_layer_size, max_layer_size // 2)),
                    dropout_cls(dropout),
                    activation,
                    self._init_weights(nn.Linear(max_layer_size // 2, latent_size))
                )

            self.decoder = nn.Sequential(
                self._init_weights(nn.Linear(latent_size, max_layer_size // 2)),
                dropout_cls(dropout),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size)),
                dropout_cls(dropout),
                activation,
                self._init_weights(nn.Linear(max_layer_size, num_features))
            )

        elif num_layers == 3:
            if encoder_type == 'concrete':
                self.encoder = ConcreteEncoder(
                    input_size=num_features,
                    K=latent_size,
                )
            else:
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation,
                    self._init_weights(nn.Linear(max_layer_size, max_layer_size // 2)),
                    dropout_cls(dropout),
                    activation,
                    self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size // 4)),
                    dropout_cls(dropout),
                    activation,
                    self._init_weights(nn.Linear(max_layer_size // 4, latent_size))
                )

            self.decoder = nn.Sequential(
                self._init_weights(nn.Linear(latent_size, max_layer_size // 4)),
                dropout_cls(dropout),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 4, max_layer_size // 2)),
                dropout_cls(dropout),
                activation,
                self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size)),
                dropout_cls(dropout),
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
        self.corruption_prob = corruption_prob

        self.save_hyperparameters()

    def _init_weights(self, layer):
        if self.activation != "selu":
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(self.activation))

        return layer

    def forward_encoder(self, x, is_training=False):
        if self.encoder_type == 'concrete':
            return self.encoder(x, training=is_training)
        else:
            return self.encoder(x)

    def forward(self, x, is_training=False):
        return self.decoder(self.forward_encoder(x, is_training=is_training))

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

    def _contractive_loss(self, mse, lambda_=1e-4, reduction=torch.sum):
        # https://github.com/AlexPasqua/Autoencoders/blob/main/src/custom_losses.py
        # https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
        # Lambda is the regularization parameter
        # reduction can be torch.mean or torch.sum
        encoder_units = reversed([child for child in self.children() if isinstance(child, nn.Linear)])
        encoder_weights = [child.weight for child in encoder_units]
        contractive_loss = torch.mean(mse + (lambda_ * torch.norm(torch.chain_matmul(*encoder_weights))), 0)
        return reduction(contractive_loss)

    def corrupt(self, x):
        # Half corruption with zeroes
        x = x * (torch.rand_like(x) > self.corruption_prob / 2).float()
        # Half corruption with random noise added
        # Random noise per column:
        noise = torch.rand_like(x) * torch.std(x, dim=0)
        torch.masked_fill(x, (torch.rand_like(x) > self.corruption_prob / 2).bool(), noise)
        return x

    def process_batch(self, batch, is_training=False):
        x, y = batch

        if self.corruption_prob > 0:
            x = self.corrupt(x)

        y_hat = self.forward(x, is_training=is_training)
        loss = F.mse_loss(y_hat, y)

        if self.encoder_type == "sparse" and is_training:
            # https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/
            sparsity = self._sparse_loss(x)
            beta = 0.001  # Weight for the sparsity penalty
            loss += beta * sparsity
        elif self.encoder_type == "contractive" and is_training:
            loss = self._contractive_loss(loss)

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
