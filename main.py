import os.path as osp

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging
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
        accelerator='cpu',
        batch_size=64
):
    kwargs = dict()
    if not use_wandb:
        kwargs['mode'] = 'disabled'
    wandb_logger = WandbLogger(project='mtProt',
                               name='AutoEncoder',
                               #log_model=use_wandb,
                               **kwargs)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/autoencoder",
        monitor="val_loss",
        verbose=True,
        save_top_k=1,
        mode="min",
        save_last=False,
        auto_insert_metric_name=True
    )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval='epoch')
    ]

    if not use_wandb:  # Sweeps use hyperband, so we don't need early stopping
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            verbose=True,
            check_on_train_epoch_end=True
        ))
    if stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging(swa_lr))

    uk_biobank = UkBioBankDataModule(batch_size=batch_size)
    uk_biobank.prepare_data()
    uk_biobank.setup(stage='fit')

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=accelerator,
        devices=1,
        auto_select_gpus=True,
        max_epochs=100,
        enable_checkpointing=True,
        default_root_dir='checkpoints/autoencoder',
        callbacks=callbacks,
        gradient_clip_val=1.0,
        detect_anomaly=False,
        fast_dev_run=False,
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

    wandb_logger.watch(model)

    trainer.fit(model, datamodule=uk_biobank)

    print("=====TRAINING COMPLETED=====")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best model val_loss: {checkpoint_callback.best_model_score}")

    print("=====TESTING=====")
    uk_biobank.setup(stage='test')
    trainer.test(ckpt_path="best", datamodule=uk_biobank)


def sweep_func():
    wandb.init(allow_val_change=True)  # (config=default_config)

    # Sanity check configuration for mutually exclusive parameters
    optimizer = wandb.config.optimizer
    if optimizer not in ('sgd', 'rmsprop'):
        wandb.config.update({'momentum': 0.0})

    if optimizer not in ('adam', 'adamw'):
        wandb.config.update({'amsgrad': 0})

    if wandb.config.swa_enabled == 0:
        wandb.config.update({'swa_lr': 0.0})

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
        accelerator='gpu',
        batch_size=512
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
    # TODO: create a scheduler for creating a downstream prediction task loss
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging errors
    run_model(accelerator='gpu', use_wandb=False, autoencoder_type='vae')


if __name__ == "__main__":
    main()
