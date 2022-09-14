import os.path as osp

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, \
    DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets import UkBioBankDataModule
from models import MtEncoder

sweep_config = {
    'method': 'random',  # bayes
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
    },
    'parameters': {
        'stochastic_weight_averaging': {
            'parameters': {
                'enabled': {
                    'values': [0, 1],
                },
                'swa_lr': {
                    'min': 0.0001,
                    'max': 0.1,
                }
            }
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'adamw', 'adamax', 'radam', 'rmsprop'],
        },
        'learning_rate': {
            'min': 0.00001,
            'max': 0.1,
        },
        'momentum': {
            'values': [0.0, 0.9, 0.99],
        },
        'weight_decay': {
            'min': 0.0,
            'max': 0.1
        },
        'amsgrad': {
            'values': [0, 1],
        },
        'num_layers': {
            'values': [1, 2, 3],
        },
        'max_layer_size': {
            'values': [64, 128, 256],
        },
        'latent_size': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 64,
        },
        'activation': {
            'values': ['relu', 'leaky_relu', 'gelu', 'selu'],
        },
        'autoencoder_type': {
            'values': [
                'vanilla',  # Standard
                'sparse',  # See Goodfellow et al. 2016, forces sparsity in the latent space to make features more interpretable
                # See also https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
                'contractive',  # See Goodfellow et al 2016, forces the latent space to be smooth to make features more interpretable
                # See also https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
                'concrete',  # https://arxiv.org/abs/1901.09346  method for unsupervised feature selection
            ],
        }
    }
}
sweep = True


def run_model(
        stochastic_weight_averaging=False,

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
        callbacks.append(StochasticWeightAveraging(wandb.config.stochastic_weight_averaging.swa_lr))

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
        detect_anomaly=True
    )

    model = MtEncoder(
        num_features=uk_biobank.num_features,
        feature_names=uk_biobank.feature_names,
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
    wandb.init()

    # Sanity check configuration for mutually exclusive parameters
    optimizer = wandb.config.optimizer
    if optimizer not in ('sgd', 'rmsprop'):
        wandb.config.momentum = 0.0

    if optimizer not in ('adam', 'adamw'):
        wandb.config.amsgrad = 0

    if wandb.config.stochastic_weight_averaging.enabled == 0:
        wandb.config.stochastic_weight_averaging.swa_lr = 0.0

    run_model(
        stochastic_weight_averaging=wandb.config.stochastic_weight_averaging.enabled == 1,
        use_wandb=True
    )


def main():
    if osp.exists('wandb_token.txt'):
        with open('wandb_token.txt', 'r') as f:
            wandb_token = f.read().strip()
        wandb.login(key=wandb_token)
        if sweep:
            sweep_id = wandb.sweep(sweep_config, project='mtProt')
            print('Sweep ID: {}'.format(sweep_id))
            wandb.agent(sweep_id, function=sweep_func, count=25, project='mtProt')
            exit()

    run_model()


if __name__ == "__main__":
    main()
