import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from torch import nn
from torch.autograd import Variable


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
                 latent_size: int,
                 activation_fn: nn.Module):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.activation = activation_fn
        self.mu_encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            self.activation
        )
        self.var_encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            self.activation
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size)
        )

    def reparameterize(self, mu, var):
        eps = Variable(var.data.new(var.size()).normal_())
        z = mu + var*eps
        return z

    def kl_loss(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = torch.mean(reproduction_loss + KLD)
        return loss

    def encode(self, x):
        mu = self.mu_encoder(x)
        log_var = self.var_encoder(x)
        z = self.reparameterize(mu, torch.exp(log_var/2))
        return mu, log_var, z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var, z = self.encode(x)
        xhat = self.decode(z)
        return xhat, mu, log_var


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

        if num_layers == 1:
            if encoder_type == 'concrete':
                self.encoder = ConcreteEncoder(
                    input_size=num_features,
                    K=latent_size,
                )
            elif encoder_type == 'vae':
                self.vae_module = VariationalEncoder(
                    input_size=num_features,
                    latent_size=latent_size,
                    activation_fn=activation_fn
                )
                self._init_weights(self.vae_module)
                self.encoder = nn.Identity()
            else:
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, latent_size)),
                    activation_fn
                )

            if encoder_type == 'vae':  # VAE has a built in decoder layer
                self.decoder = nn.Identity()
            else:
                self.decoder = nn.Sequential(
                    self._init_weights(nn.Linear(latent_size, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, num_features))
                )

        elif num_layers == 2:
            if encoder_type == 'concrete':
                self.encoder = ConcreteEncoder(
                    input_size=num_features,
                    K=latent_size,
                )
            elif encoder_type == 'vae':
                self.vae_module = VariationalEncoder(
                        input_size=max_layer_size,
                        latent_size=latent_size,
                        activation_fn=activation_fn
                    )
                self._init_weights(self.vae_module)
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn
                )
            else:
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, max_layer_size // 2)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size // 2, latent_size)),
                    activation_fn
                )

            if encoder_type == 'vae':  # VAE has a built in decoder layer
                self.decoder = nn.Sequential(
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, num_features))
                )
            else:
                self.decoder = nn.Sequential(
                    self._init_weights(nn.Linear(latent_size, max_layer_size // 2)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, num_features))
                )

        elif num_layers == 3:
            if encoder_type == 'concrete':
                self.encoder = ConcreteEncoder(
                    input_size=num_features,
                    K=latent_size,
                )
            elif encoder_type == 'vae':
                self.vae_module = VariationalEncoder(
                        input_size=max_layer_size//2,
                        latent_size=latent_size,
                        activation_fn=activation_fn
                    )
                self._init_weights(self.vae_module)
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, max_layer_size//2)),
                    dropout_cls(dropout),
                    activation_fn
                )
            else:
                self.encoder = nn.Sequential(
                    self._init_weights(nn.Linear(num_features, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, max_layer_size // 2)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size // 4)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size // 4, latent_size)),
                    activation_fn
                )

            if encoder_type == 'vae':  # VAE has a built in decoder layer
                self.decoder = nn.Sequential(
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size//2, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, num_features))
                )
            else:
                self.decoder = nn.Sequential(
                    self._init_weights(nn.Linear(latent_size, max_layer_size // 4)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size // 4, max_layer_size // 2)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size // 2, max_layer_size)),
                    dropout_cls(dropout),
                    activation_fn,
                    self._init_weights(nn.Linear(max_layer_size, num_features))
                )

        else:
            raise ValueError("num_layers must be 1, 2, or 3")

        # Metrics
        self.r2_score = torchmetrics.R2Score(num_outputs=num_features,
                                             multioutput="raw_values")
        self.mse_score = torchmetrics.MeanSquaredError()
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
        if type(layer) == VariationalEncoder:
            # Encode the linear layers
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

        if self.encoder_type == 'vae' and is_training:
            return reconstruction, mu, log_var
        return reconstruction

    def _kl_divergence(self, rho_pred, rho=0.05):
        # Rho is the sparsity parameter
        rho_hat = torch.mean(torch.sigmoid(rho_pred), dim=1)  # Convert to probabilities
        rho = torch.ones_like(rho_hat) * rho
        return torch.sum(
            rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )

    def _get_relevant_children(self, include_decoder=True):
        relevant_children = []
        for child in self.encoder.children():
            if type(child) not in (nn.AlphaDropout, nn.Dropout, nn.Identity):
               yield child
        if include_decoder:
            for child in self.decoder.children():
                if type(child) not in (nn.AlphaDropout, nn.Dropout, nn.Identity):
                    yield child

    def _sparse_loss(self, values, rho=0.05):
        # Rho is the sparsity parameter
        loss = 0
        for child in self._get_relevant_children():
            values = child(values)
            loss += self._kl_divergence(values, rho=rho)
        return loss

    def _contractive_loss(self, mse, lambda_=1e-4, reduction=torch.sum):
        # https://github.com/AlexPasqua/Autoencoders/blob/main/src/custom_losses.py
        # https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
        # Lambda is the regularization parameter
        # reduction can be torch.mean or torch.sum
        encoder_units = reversed([child for child in self._get_relevant_children(include_decoder=False) if isinstance(child, nn.Linear)])
        encoder_weights = [child.weight for child in encoder_units]
        contractive_loss = torch.mean(mse + (lambda_ * torch.norm(torch.linalg.multi_dot(encoder_weights))), 0)
        return reduction(contractive_loss)

    def corrupt(self, x):
        # Half corruption with zeroes
        x = x * (torch.rand_like(x) > self.corruption_prob / 2).float()
        # Half corruption with random noise added
        # Random noise per column:
        noise = torch.rand_like(x) * torch.std(x, dim=0)
        torch.masked_scatter(x, (torch.rand_like(x) > self.corruption_prob / 2).bool(), noise)
        return x

    def process_batch(self, batch, is_training=False):
        x, y = batch

        if self.corruption_prob > 0:
            x = self.corrupt(x)

        y_hat = self.forward(x, is_training=is_training)
        if self.encoder_type == 'vae' and is_training:
            y_hat, mu, log_var = y_hat
            loss = self.vae_module.kl_loss(y, y_hat, mu, log_var).to(y.device)
            #self.print("!2: ", loss.device)
        else:
            loss = F.mse_loss(y_hat, y)
        #self.print("!3: ", y_hat.device)

        if self.encoder_type == "sparse" and is_training:
            # https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/
            sparsity = self._sparse_loss(x)
            beta = 0.001  # Weight for the sparsity penalty
            loss += beta * sparsity
        elif self.encoder_type == "contractive" and is_training:
            loss = self._contractive_loss(loss)

        r2 = self.r2_score(y_hat, y)
        mse = self.mse_score(y_hat, y)
        r2_table = wandb.Table(data=[[feat, r2_val] for (feat, r2_val) in zip(self.feature_names, r2)],
                               columns=["feature", "r2"])
        return loss, mse, r2_table

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.define_metric('train_r2', summary='mean', goal='maximize')
            wandb.define_metric('val_r2', summary='mean', goal='maximize')
            wandb.define_metric('test_r2', summary='mean', goal='maximize')

        loss, mse, r2 = self.process_batch(batch, is_training=True)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_mse", mse, on_step=True, on_epoch=False)

        self.logger.log_metrics({"train_r2":
            wandb.plot.bar(
                r2, "feature", "r2",
                title="R2 Score by Feature"
            )}, step=self.trainer.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, r2 = self.process_batch(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True)

        self.logger.log_metrics({"val_r2":
            wandb.plot.bar(
                r2, "feature", "r2",
                title="R2 Score by Feature"
            )}, step=self.trainer.global_step)

        return loss

    def test_step(self, batch, batch_idx):
        loss, mse, r2 = self.process_batch(batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mse", mse, on_step=False, on_epoch=True)

        self.logger.log_metrics({"test_r2":
            wandb.plot.bar(
                r2, "feature", "r2",
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
