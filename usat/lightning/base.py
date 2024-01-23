import math
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from self_sup.utils.builder import DATASET, MODEL, TASK

@TASK.register_module()
class BaseTask(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.training_cfg = self.hparams["training"]
        self.batch_size = self.training_cfg["batch_size"]


    def forward(self, x):
        # TODO
        raise NotImplementedError


    def training_step(self, batch, batch_nb):
        # TODO
        raise NotImplementedError


    def validation_step(self, batch, batch_nb):
        # TODO
        raise NotImplementedError


    def training_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Train/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)


    def validation_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Val/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)


    def test_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Test/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer_name = self.training_cfg.get("optimizer", "Adam")
        optimizer_cfg = self.training_cfg.get("optimizer_cfg", {'lr': 0.001})

        if 'lr' not in optimizer_cfg:
            raise KeyError("You must provide learning rate in optimizer cfg")
        
        if optimizer_name == 'Adam':
            optimizer_class = optim.Adam
        elif optimizer_name == "SGD":
            optimizer_class = optim.SGD
        elif optimizer_name == "AdamW":
            optimizer_class = optim.AdamW
        else:
            raise ValueError(f"{optimizer_name} is not supported, add it to configure_optimizers in base lightning class.")

        optimizer = optimizer_class(self.parameters(), **optimizer_cfg)
        
        return optimizer


    def train_dataloader(self):
        dataset = DATASET.build(self.hparams, split='train')
        collate_fn = dataset.collate_fn  if hasattr(dataset, 'collate_fn') else None
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.training_cfg['train_loader_worker'],
                                 collate_fn=collate_fn)
        return data_loader


    def val_dataloader(self):
        dataset = DATASET.build(self.hparams, split='eval')
        collate_fn = dataset.collate_fn  if hasattr(dataset, 'collate_fn') else None
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.training_cfg['eval_loader_worker'],
                                 collate_fn=collate_fn)
        return data_loader

    def test_dataloader(self):
        dataset = DATASET.build(self.hparams, split='test')
        collate_fn = dataset.collate_fn  if hasattr(dataset, 'collate_fn') else None
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.training_cfg['eval_loader_worker'],
                                 collate_fn=collate_fn)
        return data_loader

@TASK.register_module()
class BasePretrainTask(BaseTask):
    def __init__(self, params):
        super().__init__(params)


@TASK.register_module()
class BaseDownstreamTask(BaseTask):
    def __init__(self, params):
        super().__init__(params)
        models = MODEL.build(params)
        self.encoder = models['encoder']
        self.head = models['head']
        gpus = params['training']['trainer']['gpus']
        self.num_gpu = gpus if isinstance(gpus, int) else len(gpus)
        self.epoch_progress = 1
        
    def train_dataloader(self):
        """overwrite to calculate epoch progress"""
        data_loader = super().train_dataloader()
        self.num_steps_per_train_epoch = math.ceil(len(data_loader) / self.num_gpu)
        return data_loader

    def configure_optimizers(self):
        """ Configure optimizer and scheduler """
        optimizer = super().configure_optimizers()
        self.scheduler_name = self.training_cfg.get("scheduler", None)
        scheduler_cfg = self.training_cfg.get("scheduler_cfg", {})

        # Only ReduceLROnPlateau operates on epoch interval
        if self.scheduler_name == "Plateau":
            monitor_metric = scheduler_cfg.pop("monitor_metric", "Val/epoch_avg_loss")
            scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
            return [optimizer], [{"scheduler": scheduler_class, 
                                  "monitor": monitor_metric}]
        elif self.scheduler_name == "MultiStep":
            scheduler_class = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg)
            return [optimizer], [{"scheduler": scheduler_class,
                                  "interval": "epoch"}]
        # Manually schedule for Cosine and Polynomial scheduler
        elif self.scheduler_name in [None, "Cosine", "Poly"]:
            return optimizer
        else:
            raise ValueError(f"{self.scheduler_name} is not supported, add it to configure_optimizers in BaseDownstreamTask.")


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """Overwrite with warmup epoch and manual lr decay"""

        self.epoch_progress = self.current_epoch + min((batch_idx+1)/self.num_steps_per_train_epoch , 1)
        initial_lr = self.training_cfg["optimizer_cfg"]['lr']
        warm_up_epoch = self.training_cfg.get("warm_up_epoch", 0)
        max_epochs = self.training_cfg['trainer']['max_epochs']

        if self.scheduler_name in ["Cosine", "Poly"] or self.epoch_progress <= warm_up_epoch:
            if self.epoch_progress <= warm_up_epoch:
                lr = initial_lr * self.epoch_progress / warm_up_epoch
            elif self.scheduler_name == "Cosine":
                lr = initial_lr * 0.5 * (1. + math.cos(math.pi * (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)))
            else:
                power = self.training_cfg.get("scheduler_cfg", {}).get("power", 0.5)
                lr = initial_lr * (1. - (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)) ** power
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr    

        optimizer.step(closure=optimizer_closure)