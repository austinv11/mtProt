import os.path as osp

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, \
    DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger

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
                    'values': [True, False],
                },
                'swa_lr': {
                    'min': 0.0001,
                    'max': 0.1,
                }
            }
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

    )

    wandb_logger.watch(model, log='all')

    trainer.fit(model, train_dataloaders=None, val_dataloaders=None)

    print("=====TRAINING COMPLETED=====")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best model val_loss: {checkpoint_callback.best_model_score}")

    print("=====TESTING=====")
    trainer.test(ckpt_path="best", dataloaders=get_sequence_loader(DatasetMode.TEST, window_size=window_size))


def sweep_func():
    wandb.init()
    run_model(
        stochastic_weight_averaging=wandb.config.stochastic_weight_averaging.enabled,
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
