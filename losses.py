from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class MultiTaskLossScheduler(ABC):

    @abstractmethod
    def loss(self, epoch: int, max_epochs: int, ae_loss: torch.Tensor, regression_loss: torch.Tensor) -> torch.Tensor:
        ...


class AutoEncoderLossOnly(MultiTaskLossScheduler):

    def loss(self, epoch: int, max_epochs: int, ae_loss: torch.Tensor, regression_loss: torch.Tensor) -> torch.Tensor:
        return ae_loss


class RegressionLossOnly(MultiTaskLossScheduler):

    def loss(self, epoch: int, max_epochs: int, ae_loss: torch.Tensor, regression_loss: torch.Tensor) -> torch.Tensor:
        return regression_loss


class SumLoss(MultiTaskLossScheduler):

    def __init__(self, ae_loss_weight=1.0, regression_loss_weight=1.0):
        self.ae_loss_weight = ae_loss_weight
        self.regression_loss_weight = regression_loss_weight

    def loss(self, epoch: int, max_epochs: int, ae_loss: torch.Tensor, regression_loss: torch.Tensor) -> torch.Tensor:
        return self.ae_loss_weight * ae_loss + self.regression_loss_weight * regression_loss


class IntersectingLoss(MultiTaskLossScheduler):

    def __init__(self, vae_warmup_epochs: int, freeze_after_epochs: int = None, min_regression_weight: float = 0, max_regression_weight: float = 1):
        self.vae_warmup_epochs = vae_warmup_epochs  # Only consider VAE loss for the first few epochs
        self.freeze_after_epochs = freeze_after_epochs  # Stop modifying loss formula after these epochs
        self.min_regression_weight = min_regression_weight  # Min and max weight for the regression loss
        self.max_regression_weight = max_regression_weight

    def loss(self, epoch: int, max_epochs: int, ae_loss: torch.Tensor, regression_loss: torch.Tensor) -> torch.Tensor:
        if epoch < self.vae_warmup_epochs:
            return ae_loss
        max_effective_epoch = max_epochs - self.vae_warmup_epochs - (self.freeze_after_epochs or 0)
        effective_epoch = min(epoch - self.vae_warmup_epochs, max_effective_epoch)
        progress = effective_epoch / max_effective_epoch
        regression_weight = self.min_regression_weight + progress * (self.max_regression_weight - self.min_regression_weight)
        vae_weight = 1 - regression_weight
        return vae_weight * ae_loss + regression_weight * regression_loss


def vae_kl_loss(x_hat, x, mean, log_var, kl_beta=1e-2):
    reproduction_loss = F.mse_loss(x_hat, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
    return torch.mean(reproduction_loss + kl_beta * KLD)


def sparse_kl_divergence(rho_pred, rho=0.05):
    # Rho is the sparsity parameter
    rho_hat = torch.mean(torch.sigmoid(rho_pred), dim=1)  # Convert to probabilities
    rho = torch.ones_like(rho_hat) * rho
    return torch.sum(
        rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


def _get_relevant_children(model: 'MtEncoder', include_decoder=True):
    for child in model.encoder.children():
        if type(child) not in (nn.AlphaDropout, nn.Dropout, nn.Identity):
           yield child
    if include_decoder:
        for child in model.decoder.children():
            if type(child) not in (nn.AlphaDropout, nn.Dropout, nn.Identity):
                yield child


def sparse_loss(model: 'MtEncoder', values, rho=0.05):
    # Rho is the sparsity parameter
    loss = 0
    for child in _get_relevant_children(model, include_decoder=True):
        values = child(values)
        loss += sparse_kl_divergence(values, rho=rho)
    return loss


def contractive_loss(model: 'MtEncoder', mse, lambda_=1e-4, reduction=torch.sum):
    # https://github.com/AlexPasqua/Autoencoders/blob/main/src/custom_losses.py
    # https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
    # Lambda is the regularization parameter
    # reduction can be torch.mean or torch.sum
    encoder_units = reversed([child for child in _get_relevant_children(model, include_decoder=False) if isinstance(child, nn.Linear)])
    encoder_weights = [child.weight for child in encoder_units]
    contractive_loss = torch.mean(mse + (lambda_ * torch.norm(torch.linalg.multi_dot(encoder_weights))), 0)
    return reduction(contractive_loss)
