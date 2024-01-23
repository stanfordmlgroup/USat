import uuid
from pathlib import Path

import fire
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DataParallelStrategy, DDPStrategy

from usat.core.serialization import read_yaml
from usat.utils.builder import TASK
from usat.utils.constants import WANDB_ENTITY, WANDB_PROJECT


def train(cfg):
    cfg = read_yaml(cfg)
    trainer_cfg = cfg['training']['trainer']

    random_seed = cfg.get('random_seed')
    if random_seed:
        pl.seed_everything(random_seed)

    # get task
    task = TASK.build(cfg)
    
    # GPU
    gpus = trainer_cfg.get('gpus', 1)
    strategy = trainer_cfg.get('strategy', None)
    unused_params = trainer_cfg.get('unused_params', False)

    num_gpu = gpus if isinstance(gpus, int) else len(gpus)
    if gpus == -1 or num_gpu > 1:
        if strategy == 'dp':
            strategy = DataParallelStrategy()
        if strategy == 'ddp':
            strategy = DDPStrategy(find_unused_parameters=unused_params)
    else:
        strategy = None

    # Precision
    mixed_precision = cfg.get("mixed_precision", False)
    if mixed_precision:
        precision = 16
    else:
        precision = 32
    
    # Logging
    save_dir = trainer_cfg.get('save_dir', 'results/default')
    exp_name = cfg.get('exp_name', f'exp_{str(uuid.uuid1())[:8]}') 
    version = cfg.get('version', 0)
    wandb_entity = cfg.get('w&b_entity', WANDB_ENTITY)
    wandb_project = cfg.get('w&b_project', WANDB_PROJECT)

    logger = WandbLogger(name=f'{exp_name}_v{version}', save_dir=save_dir,
                         project=wandb_project, entity=wandb_entity)

    # Callbacks - Checkpointing and early stop
    save_top_k = trainer_cfg.get('save_top_k', 5)
    monitor_metric = trainer_cfg.get('monitor_metric', 'Eval_Loss')
    monitor_mode = trainer_cfg.get('monitor_mode', 'min')
    patience = trainer_cfg.get('patience', 10)

    ckpt_dir = Path(save_dir)/ exp_name/ f'version_{version}'/ 'ckpt'
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=save_top_k, 
                              verbose=True, monitor=monitor_metric, 
                              mode=monitor_mode, every_n_epochs=1)
    earlystop_cb = EarlyStopping(monitor=monitor_metric, 
                                 patience=patience, 
                                 verbose=True, mode=monitor_mode)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer config
    gradient_clip_val = trainer_cfg.get('gradient_clip_val', 0)
    limit_train_batches = trainer_cfg.get('limit_train_batches', 1.0)
    enable_model_summary = trainer_cfg.get('enable_model_summary', False)
    max_epochs = trainer_cfg.get('max_epochs', 100)
    accumulated_batches = trainer_cfg.get('accumulated_batches', 1)
    resume_checkpoint = trainer_cfg.get('resume_ckpt', None)

    trainer = Trainer(accelerator="gpu",
                      precision=precision,
                      devices=gpus,
                      strategy=strategy,
                      logger=logger,
                      callbacks=[ckpt_cb, earlystop_cb, lr_monitor],
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      enable_model_summary = enable_model_summary,
                      max_epochs=max_epochs,
                      accumulate_grad_batches=accumulated_batches,
                      log_every_n_steps=5)
    
    if resume_checkpoint:
        trainer.fit(task, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(task)


def test(cfg):
    cfg = read_yaml(cfg)
    trainer_cfg = cfg['training']['trainer']

    random_seed = cfg.get('random_seed')
    if random_seed:
        pl.seed_everything(random_seed)

    # get task
    task = TASK.build(cfg)
    
    # GPU
    gpus = trainer_cfg.get('gpus', 1)
    strategy = trainer_cfg.get('strategy', None)
    unused_params = trainer_cfg.get('unused_params', False)

    num_gpu = gpus if isinstance(gpus, int) else len(gpus)
    if gpus == -1 or num_gpu > 1:
        if strategy == 'dp':
            strategy = DataParallelStrategy()
        if strategy == 'ddp':
            strategy = DDPStrategy(find_unused_parameters=unused_params)
    else:
        strategy = None

    # Precision
    mixed_precision = cfg.get("mixed_precision", False)
    if mixed_precision:
        precision = 16
    else:
        precision = 32
 
    # Trainer config
    gradient_clip_val = trainer_cfg.get('gradient_clip_val', 0)
    limit_train_batches = trainer_cfg.get('limit_train_batches', 1.0)
    enable_model_summary = trainer_cfg.get('enable_model_summary', False)
    max_epochs = trainer_cfg.get('max_epochs', 100)
    accumulated_batches = trainer_cfg.get('accumulated_batches', 1)
    test_checkpoint = trainer_cfg.get('test_ckpt', None)

    trainer = Trainer(accelerator="gpu",
                      precision=precision,
                      devices=1,
                      num_nodes=1,
                      strategy=strategy,
                      logger=False,
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      enable_model_summary = enable_model_summary,
                      max_epochs=max_epochs,
                      accumulate_grad_batches=accumulated_batches,
                      log_every_n_steps=5)
    
    trainer.test(task, ckpt_path=test_checkpoint)

if __name__ == "__main__":
    fire.Fire()
