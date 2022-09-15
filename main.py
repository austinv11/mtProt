import os.path as osp

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, \
    DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets import UkBioBankDataModule
from models import MtEncoder

default_config = dict(
    swa_enabled=1,
    swa_lr=0.05,
    optimizer='adamw',
    learning_rate=0.001,
    weight_decay=0,
    momentum=0,
    amsgrad=0,
    num_layers=2,
    dropout=0,
    corruption_prob=0,
    max_layer_size=120,
    latent_size=32,
    activation='relu',
    autoencoder_type='vanilla'
)


def run_model(
        latent_size=32,
        stochastic_weight_averaging=False,
        swa_lr=0.05,
        optimizer='adam',
        learning_rate=0.001,
        momentum=0.9,
        amsgrad=False,
        weight_decay=0.0,
        num_layers=2,
        max_layer_size=64,
        activation='relu',
        autoencoder_type='vanilla',
        dropout=0.0,
        corruption_prob=0.0,

        use_wandb=False,
        accelerator='cpu'
):
    kwargs = dict()
    if use_wandb:
        kwargs['mode'] = 'disabled'
    wandb_logger = WandbLogger(project='mtProt', name='AutoEncoder', log_model=True, **kwargs)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/autoencoder",
        monitor="val_loss",
        verbose=True,
        save_top_k=3,
        mode="min",
        save_last=True,
        auto_insert_metric_name=True
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            verbose=True,
            min_delta=0.0001
        ),
        LearningRateMonitor(logging_interval='step'),
        DeviceStatsMonitor()
    ]

    if stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging(swa_lr))

    uk_biobank = UkBioBankDataModule()
    uk_biobank.prepare_data()
    uk_biobank.setup(stage='fit')

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=accelerator,
        devices=1,
        auto_select_gpus=True,
        max_epochs=250,
        enable_checkpointing=True,
        default_root_dir='checkpoints/autoencoder',
        callbacks=callbacks,
        gradient_clip_val=1.0,
        detect_anomaly=True,
        fast_dev_run=True,
    )

    model = MtEncoder(
        num_features=uk_biobank.num_features,
        feature_names=uk_biobank.feature_names,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        num_layers=num_layers,
        max_layer_size=max_layer_size,
        latent_size=latent_size,
        activation=activation,
        optimizer=optimizer,
        encoder_type=autoencoder_type,
        dropout=dropout,
        corruption_prob=corruption_prob
    )

    wandb_logger.watch(model, log='all')

    trainer.fit(model, datamodule=uk_biobank)

    print("=====TRAINING COMPLETED=====")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best model val_loss: {checkpoint_callback.best_model_score}")

    print("=====TESTING=====")
    uk_biobank.setup(stage='test')
    trainer.test(ckpt_path="best", datamodule=uk_biobank)


def sweep_func():
    wandb.init(config=default_config)

    # Sanity check configuration for mutually exclusive parameters
    optimizer = wandb.config.optimizer
    if optimizer not in ('sgd', 'rmsprop'):
        wandb.config.update({'momentum': 0.0}, allow_val_change=True)

    if optimizer not in ('adam', 'adamw'):
        wandb.config.update({'amsgrad': 0}, allow_val_change=True)

    if wandb.config.swa_enabled == 0:
        wandb.config.update({'swa_lr': 0.0}, allow_val_change=True)

    run_model(
        latent_size=wandb.config.latent_size,
        stochastic_weight_averaging=wandb.config.swa_enabled == 1,
        swa_lr=wandb.config.swa_lr,
        optimizer=wandb.config.optimizer,
        momentum=wandb.config.momentum,
        amsgrad=wandb.config.amsgrad == 1,
        weight_decay=wandb.config.weight_decay,
        num_layers=wandb.config.num_layers,
        max_layer_size=wandb.config.max_layer_size,
        activation=wandb.config.activation,
        autoencoder_type=wandb.config.autoencoder_type,
        dropout=wandb.config.dropout,
        corruption_prob=wandb.config.corruption_prob,

        use_wandb=True,
        accelerator='cpu'
    )


def main():
    if osp.exists('wandb_token.txt'):
        with open('wandb_token.txt', 'r') as f:
            wandb_token = f.read().strip()
        wandb.login(key=wandb_token)
#        if sweep:
#            sweep_id = wandb.sweep(sweep_config, project='mtProt')
#            print('Sweep ID: {}'.format(sweep_id))
#            wandb.agent(sweep_id, function=sweep_func, count=25, project='mtProt')
#            exit()
#        else:
        sweep_func()
        exit()

    run_model()


if __name__ == "__main__":
    main()
