import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from torch import nn
from torch.autograd import Variable

from losses import vae_kl_loss, sparse_loss, contractive_loss, MultiTaskLossScheduler, AutoEncoderLossOnly


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
        self.logits = nn.Parameter(torch.zeros([K, input_size]), requires_grad=True)
        self.selections = nn.Parameter(torch.zeros([input_size, K]), requires_grad=False)
        self.learn = nn.Parameter(torch.tensor(1), requires_grad=False)

        nn.init.xavier_normal_(self.logits)

    @classmethod
    def calculate_alpha(cls, num_epochs: int, features: int, batch_size: int, start_temp: float = 10.0, end_temp: float = 0.1):
        # Calculate ideal alpha for the concrete encoder
        # Ref: https://github.com/mfbalin/Concrete-Autoencoders/blob/master/concrete_autoencoder/concrete_autoencoder/__init__.py#L89
        return math.exp(math.log(end_temp / start_temp) / num_epochs * (features + batch_size - 1) // batch_size)

    def forward(self, x, training=False):
        self.temp.data = torch.clamp(self.temp.data * self.alpha, self.end_temp, self.start_temp)
        noisy_logits = (self.logits + self.gumbel_dist.sample(self.logits.shape).to(x.device)) / self.temp
        samples = F.softmax(noisy_logits, dim=0)
        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=0), self.logits.shape[0])

        if training:
            self.selections.data = samples.T
        else:
            self.selections.data = discrete_logits.float()

        return x @ self.selections


# Variational AutoEncoder
# Does not implement the hidden layers, only the latent space transformation
# Based on https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class VariationalEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 latent_size: int):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.mu_encoder = nn.Sequential(
            nn.Linear(input_size, latent_size)
        )
        self.var_encoder = nn.Sequential(
            nn.Linear(input_size, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size)
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x):
        mu = self.mu_encoder(x)
        log_var = self.var_encoder(x)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var, z = self.encode(x)
        xhat = self.decode(z)
        return xhat, mu, log_var


class MtEncoder(pl.LightningModule):

    def __init__(self,
                 num_input: int,
                 input_names: list[str],
                 num_output: int,
                 output_names: list[str],
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
                 corruption_prob: float = 0.0,
                 loss_scheduler: MultiTaskLossScheduler = AutoEncoderLossOnly(),
                 ):
        super().__init__()

        self.activation = activation
        self.loss_scheduler = loss_scheduler
        self.dropout = dropout

        if dropout == 0:
            dropout_cls = nn.Identity
        elif activation == "selu":
            dropout_cls = nn.AlphaDropout
        else:
            dropout_cls = nn.Dropout

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            activation_fn = nn.LeakyReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "selu":
            activation_fn = nn.SELU()
        else:
            raise ValueError(f"Unknown activation function {activation}")

        linear_layer_latent = max_layer_size//(2**(num_layers-1)) if encoder_type == "vae" else latent_size
        if encoder_type == "concrete":
            self.encoder = ConcreteEncoder(
                input_size=num_input,
                K=latent_size,
            )
        else:
            self.encoder = self._make_encoder(
                latent_size=linear_layer_latent,
                num_features=num_input,
                num_layers=num_layers,
                max_layer_size=max_layer_size,
                dropout_cls=dropout_cls,
                activation_fn=activation_fn
            )
            self._init_weights(self.encoder)

        if encoder_type == "vae":
            self.vae_module = VariationalEncoder(
                input_size=linear_layer_latent,
                latent_size=latent_size
            )
            self._init_weights(self.vae_module)

        self.decoder = self._make_decoder(
            latent_size=latent_size,
            num_features=num_input,
            num_layers=num_layers,
            max_layer_size=max_layer_size,
            dropout_cls=dropout_cls,
            activation_fn=activation_fn
        )
        self._init_weights(self.decoder)

        self.regression_decoder = self._make_decoder(
            latent_size=latent_size,
            num_features=num_output,
            num_layers=num_layers,
            max_layer_size=max_layer_size,
            dropout_cls=dropout_cls,
            activation_fn=activation_fn
        )
        self._init_weights(self.regression_decoder)

        # Metrics
        self.r2_score = torchmetrics.R2Score(num_outputs=num_input,
                                             multioutput="raw_values")
        self.mse_score = torchmetrics.MeanSquaredError()
        self.feature_names = input_names
        self.target_names = output_names
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.encoder_type = encoder_type
        self.corruption_prob = corruption_prob

        self.save_hyperparameters()

    def _make_decoder(self,
                      latent_size,
                      num_features,
                      num_layers,
                      max_layer_size,
                      dropout_cls,
                      activation_fn):
        layers = []
        for i in range(num_layers-1, -1, -1):
            in_size = latent_size if i == num_layers-1 else max_layer_size // (2 ** (i))
            out_size = num_features if i == 0 else max_layer_size // (2 ** (i-1))
            layers.append(
                self._init_weights(nn.Linear(in_size, out_size))
            )
            if i != 0:
                layers.append(dropout_cls(self.dropout))
                layers.append(activation_fn)
        return nn.Sequential(*layers)

    def _make_encoder(self,
                      latent_size,
                      num_features,
                      num_layers,
                      max_layer_size,
                      dropout_cls,
                      activation_fn):
        layers = []
        for i in range(num_layers):
            in_size = num_features if i == 0 else max_layer_size // (2 ** (i - 1))
            out_size = latent_size if i == num_layers - 1 else max_layer_size // (2 ** i)
            layers.append(
                self._init_weights(nn.Linear(in_size, out_size))
            )
            if i != num_layers - 1:
                layers.append(dropout_cls(self.dropout))
                layers.append(activation_fn)
        return nn.Sequential(*layers)

    def _init_weights(self, layer):
        if type(layer) == VariationalEncoder:
            self._init_weights(layer.mu_encoder)
            self._init_weights(layer.var_encoder)
            self._init_weights(layer.decoder)
        if type(layer) == nn.Sequential:
            for l in layer:
                self._init_weights(l)
        if type(layer) == nn.Linear:
            if self.activation != "selu":
                can_calc_gain = self.activation in ["relu", "leaky_relu"]
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(self.activation) if can_calc_gain else 1.0)

        return layer

    def forward_encoder(self, x, is_training=False):
        if self.encoder_type == 'concrete':
            return self.encoder(x, training=is_training)
        elif self.encoder_type == 'vae':
            hidden = self.encoder(x)
            mu, log_var, z = self.vae_module.encode(hidden)
            if is_training:
                return mu, log_var, z
            else:
                return z
        else:
            return self.encoder(x)

    def forward(self, x, is_training=False):
        z = self.forward_encoder(x, is_training=is_training)
        if self.encoder_type == 'vae':
            if is_training:
                mu, log_var, z = z
            z = self.vae_module.decode(z)

        reconstruction = self.decoder(z)
        regression = self.regression_decoder(z)

        if self.encoder_type == 'vae' and is_training:
            return regression, reconstruction, mu, log_var
        return regression, reconstruction, None, None

    def corrupt(self, x):
        # Half corruption with zeroes
        x = x * (torch.rand_like(x) > self.corruption_prob / 2).float()
        # Half corruption with random noise added
        # Random noise per column:
        noise = torch.rand_like(x) * torch.std(x, dim=0)
        torch.masked_scatter(x, (torch.rand_like(x) > self.corruption_prob / 2).bool(), noise)
        return x

    def process_batch(self, batch, is_training=False):
        max_epochs = self.trainer.max_epochs
        curr_epoch = self.current_epoch

        x, y = batch

        # Ignore nan values for loss calculation
        x_nan_mask = torch.isnan(x)
        y_nan_mask = torch.isnan(y)
        x = torch.nan_to_num(x, 0)
        y = torch.nan_to_num(y, 0)

        x_orig = x

        if self.corruption_prob > 0:
            x = self.corrupt(x)

        y_hat, x_hat, mu, log_var = self.forward(x, is_training=is_training)

        # Replace nan values with predicted values to prevent propagating loss
        x_orig = x_orig*(torch.logical_not(x_nan_mask)) + x_hat*(x_nan_mask)
        y = y*(torch.logical_not(y_nan_mask)) + y_hat*(y_nan_mask)

        if self.encoder_type == 'vae' and is_training:
            autoencoder_loss = vae_kl_loss(x_hat, x_orig, mu, log_var)
        else:
            autoencoder_loss = F.mse_loss(x_hat, x_orig)
        regression_loss = F.mse_loss(y_hat, y)

        if self.encoder_type == "sparse" and is_training:
            # https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/
            sparsity = sparse_loss(self, x_orig)
            beta = 0.001  # Weight for the sparsity penalty
            autoencoder_loss += beta * sparsity
        elif self.encoder_type == "contractive" and is_training:
            autoencoder_loss = contractive_loss(self, autoencoder_loss)

        total_loss = self.loss_scheduler.loss(curr_epoch, max_epochs, autoencoder_loss, regression_loss)

        autoencoder_r2 = self.r2_score(x_hat, x_orig)
        regression_r2 = self.r2_score(y_hat, y)
        autoencoder_mse = self.mse_score(x_hat, x_orig)
        regression_mse = self.mse_score(y_hat, y)
        autoencoder_r2_table = wandb.Table(data=[[feat, r2_val] for (feat, r2_val) in zip(self.feature_names, autoencoder_r2)],
                               columns=["feature", "r2"])
        regression_r2_table = wandb.Table(data=[[feat, r2_val] for (feat, r2_val) in zip(self.target_names, regression_r2)],
                                 columns=["feature", "r2"])
        return total_loss, autoencoder_loss, regression_loss, autoencoder_mse, autoencoder_r2_table, regression_mse, regression_r2_table

    def training_step(self, batch, batch_idx):
        loss, autoencoder_loss, regression_loss, autoencoder_mse, autoencoder_r2_table, regression_mse, regression_r2_table = self.process_batch(batch, is_training=True)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_autoencoder_loss", autoencoder_loss, on_step=True, on_epoch=False)
        self.log("train_regression_loss", regression_loss, on_step=True, on_epoch=False)
        self.log("train_autoencoder_mse", autoencoder_mse, on_step=True, on_epoch=False)
        self.log("train_regression_mse", regression_mse, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.define_metric('val_autoencoder_r2', summary='last')
            wandb.define_metric('val_regression_r2', summary='last')

        loss, autoencoder_loss, regression_loss, autoencoder_mse, autoencoder_r2_table, regression_mse, regression_r2_table = self.process_batch(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_autoencoder_loss", autoencoder_loss, on_step=False, on_epoch=True)
        self.log("val_regression_loss", regression_loss, on_step=False, on_epoch=True)
        self.log("val_autoencoder_mse", autoencoder_mse, on_step=False, on_epoch=True)
        self.log("val_regression_mse", regression_mse, on_step=False, on_epoch=True)

        self.logger.log_metrics({"val_autoencoder_r2":
            wandb.plot.bar(
                autoencoder_r2_table, "feature", "r2",
                title="R2 Score by Feature"
            )}, step=self.trainer.global_step)
        self.logger.log_metrics({"val_regression_r2":
            wandb.plot.bar(
                regression_r2_table, "feature", "r2",
                title="R2 Score by Feature"
            )}, step=self.trainer.global_step)

        return loss

    def test_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.define_metric('test_autoencoder_r2', summary='last')
            wandb.define_metric('test_regression_r2', summary='last')

        loss, autoencoder_loss, regression_loss, autoencoder_mse, autoencoder_r2_table, regression_mse, regression_r2_table = self.process_batch(batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_autoencoder_loss", autoencoder_loss, on_step=False, on_epoch=True)
        self.log("test_regression_loss", regression_loss, on_step=False, on_epoch=True)
        self.log("test_autoencoder_mse", autoencoder_mse, on_step=False, on_epoch=True)
        self.log("test_regression_mse", regression_mse, on_step=False, on_epoch=True)

        self.logger.log_metrics({"test_autoencoder_r2":
            wandb.plot.bar(
                autoencoder_r2_table, "feature", "r2",
                title="R2 Score by Feature"
            )}, step=self.trainer.global_step)
        self.logger.log_metrics({"test_regression_r2":
            wandb.plot.bar(
                regression_r2_table, "feature", "r2",
                title="R2 Score by Feature"
            )}, step=self.trainer.global_step)

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
