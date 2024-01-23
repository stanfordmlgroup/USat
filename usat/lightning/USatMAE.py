import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from .base import BasePretrainTask
from .optimizer import LARS
from usat.utils.builder import MODEL, DATASET, TASK

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)


@TASK.register_module()
class USatMAE (BasePretrainTask):
    def __init__ (self, params):
        # linearly rescale lr before passing to save final parameters
        gpus = params['training']['trainer']['gpus']
        self.num_gpu = gpus if isinstance(gpus, int) else len(gpus)
        self.batch_size = params['training']['batch_size']
        lr = params['training']['optimizer_cfg']['lr']
        params['training']['optimizer_cfg']['lr'] = lr * (self.batch_size * self.num_gpu) / 256

        super().__init__(params=params)

        models = MODEL.build(self.hparams)
        self.mae = models['mae']

        self.max_epochs = self.hparams['training']['trainer']['max_epochs']
        self.mask_ratio = self.hparams['training']['mae']['mask_ratio']
        self.spectral_mask_ratio = self.hparams['training']['mae'].get('spectral_mask_ratio', self.mask_ratio)
        self.log_img_per_steps = self.hparams['training']['mae']['log_img_per_steps']
        self.input_params = self.hparams['models']['mae']['args']['input_parmas']

        self.epoch_progress = 1


    def forward(self, x):
        """
        x1/x2: (n, c, h, w)
        """
        return self.mae(x, self.mask_ratio, self.spectral_mask_ratio)
    
    def training_step(self, batch, batch_nb):
        """
        Param:
        __________
        batch:
            Shape: (n, h, w, c)

        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        
        self.epoch_progress = self.current_epoch + batch_nb/self.num_steps_per_train_epoch

        x, _ = batch

        loss, pred, mask = self.forward(x)

        # log images
        if (batch_nb + self.num_steps_per_train_epoch * self.current_epoch) % self.log_img_per_steps  == 0:
            for (k, input_band), (_, prediction), (_, image_mask) in zip(x.items(), pred.items(), mask.items()):
                prediction = pred[k]
                image_mask = mask[k]
                num_patch = self.input_params[k]['num_patch']
                gsd = self.input_params[k]['GSD']
                p = self.mae.ground_cover // gsd // num_patch
                b = input_band.shape[0]
                num_patch = input_band.shape[-1]//p
                
                prediction = self.mae.unpatchify(prediction, p)
                image_keep = nn.functional.conv_transpose2d((image_mask==0).float().reshape(b, 1, num_patch, num_patch), torch.ones(1,1,p,p, device=image_mask.device), stride=p)
                image_mask = nn.functional.conv_transpose2d((image_mask==1).float().reshape(b, 1, num_patch, num_patch), torch.ones(1,1,p,p, device=image_mask.device), stride=p)
                input_band = input_band[0][0]
                masked_input = input_band*image_keep[0][0]
                prediction = prediction[0][0]*image_mask[0][0] + input_band*image_keep[0][0]

                self.logger.log_image(key=f"Sample/{k}", images=[input_band, masked_input, prediction], caption=['Input', 'Masked Input', 'Prediction'])

        self.log("Debug/epoch_progress", self.epoch_progress, logger=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/R0_loss", loss, logger=True, prog_bar=False, rank_zero_only=True)

        return {"loss" : loss}

    def validation_step(self, batch, batch_nb):
        """overwrite to skip for now"""
        pass


    def validation_epoch_end(self, outputs):
        """overwirte to skip for now"""
        pass


    def train_dataloader(self):
        """overwrite since we need to drop last"""
        dataset = DATASET.build(self.hparams, split='train')
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, collate_fn=dataset.collate_fn,
                                 num_workers=self.training_cfg['train_loader_worker'])
        self.num_steps_per_train_epoch = len(data_loader) // self.num_gpu
        return data_loader


    def val_dataloader(self):
        pass


    def configure_optimizers(self):
        optimizer_name = self.training_cfg.get("optimizer", "LARS")
        optimizer_cfg = self.training_cfg.get("optimizer_cfg", {'lr': 0.001})

        if 'lr' not in optimizer_cfg:
            raise KeyError("You must provide learning rate in optimizer cfg")
        
        if optimizer_name == "AdamW":
            optimizer_class = torch.optim.AdamW
        elif optimizer_name == 'LARS':
            optimizer_class = LARS
        else:
            raise ValueError(f"{optimizer_name} is not supported for MoCoV3, add it to configure_optimizers in base lightning class.")

        optimizer = optimizer_class(self.parameters(), **optimizer_cfg)
        return optimizer

    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """Decays the learning rate with half-cycle cosine after warmup"""
        initial_lr = self.hparams['training']['optimizer_cfg']['lr']
        warm_up_epoch = self.hparams['training']['warmup_epoch']
        if self.epoch_progress < warm_up_epoch:
            lr = initial_lr * self.epoch_progress / warm_up_epoch
        else:
            lr = initial_lr * 0.5 * (1. + math.cos(math.pi * (self.epoch_progress - warm_up_epoch) / (self.max_epochs - warm_up_epoch)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step(closure=optimizer_closure)
