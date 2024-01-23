import typing as T

import torch
import torch.nn as nn
import torchmetrics
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import torchmetrics.classification as tmc

from .base import BaseDownstreamTask
from usat.utils.builder import TASK
from usat.utils.constants import PROBING, FINETUNING, TRAIN_METHODS

# HACK: reverse this change when standarlization is part of the dataset
from torchvision.transforms.functional import normalize
from usat.utils.constants import FMOW_RGB_MEAN, FMOW_RGB_STD

@TASK.register_module()
class MulticlassClassification (BaseDownstreamTask):
    def __init__(self, params: T.Dict[str, T.Any]) -> None:
        super().__init__(params)
        # mixup config
        self.mixup_fn = None
        mix_cfg = self.training_cfg.get('mix_cfg', {})
        if mix_cfg.get('mixup', 0) > 0 or mix_cfg.get('cutmix', 0) > 0 or mix_cfg.get('cutmix_minmax',None):
            self.mixup_fn = Mixup(**mix_cfg)
        if self.mixup_fn:
            # smoothing is handled with mixup label transform
            self.mixup_criterion = SoftTargetCrossEntropy()

        # Set to a large number if num_classes pnot found
        num_classes = getattr(self.head, 'num_classes', 10000)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.eval_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.training_method = params.get('training_method', PROBING)
        self.feature_opt = params.get('feature_opt', 'flatten')
        assert(self.training_method in TRAIN_METHODS)

    def forward(self, x: torch.Tensor):
        # print(self.encoder.training)
        if self.training_method == PROBING:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        if self.feature_opt == 'flatten':
            features = features.flatten(1)
        elif self.feature_opt == 'cls_token':
            features = features[:, 0]
        x = self.head(features)
        return x

    
    def training_step(self, batch, batch_nb):
        if self.training_method == PROBING:
            self.encoder.eval()
        else:
            self.encoder.train()
        
        # print(self.encoder.layer1[0].conv1.weight[0][0][0][0])
        x, y = batch

        # TODO: to delete - FMOW already standarlize in dataset
        # # HACK
        # x = normalize(x, mean=FMOW_RGB_MEAN, std=FMOW_RGB_STD)

        if self.mixup_fn:
            x, y = self.mixup_fn(x,y)
            logit = self.forward(x)
            loss = self.mixup_criterion(logit, y)
        else:
            logit = self.forward(x)
            loss = self.criterion(logit, y)

        pred = logit.argmax(1)
        acc = self.train_acc(pred, y)

        self.log('Train/loss', loss, prog_bar=False)
        self.log('Train/acc', acc, prog_bar=True)
        return {'loss': loss,
                'accuracy': acc}

    
    def training_epoch_end(self, outputs):
        self.train_acc.reset()
        return super().training_epoch_end(outputs)


    def validation_step(self, batch, batch_nb):
        self.encoder.eval()
        with torch.no_grad():
            x, y = batch
            # TODO: to delete - FMOW already standarlize in dataset
            # # HACK
            # x = normalize(x, mean=FMOW_RGB_MEAN, std=FMOW_RGB_STD)
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(1)
            acc = self.eval_acc(pred, y)

        self.log('Val/loss', loss, prog_bar=False, sync_dist=True)
        self.log('Val/acc', acc, prog_bar=True, sync_dist=True)
        return {'loss': loss,
                'accuracy': acc}
    

    def validation_epoch_end(self, outputs):
        self.eval_acc.reset()
        return super().validation_epoch_end(outputs)


    def test_step(self, batch, batch_nb):
        self.encoder.eval()
        with torch.no_grad():
            x, y = batch
            # TODO: to delete - FMOW already standarlize in dataset
            # # HACK
            # x = normalize(x, mean=FMOW_RGB_MEAN, std=FMOW_RGB_STD)
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(1)
            acc = self.test_acc(pred, y)

        self.log('Test/loss', loss, prog_bar=False, sync_dist=True)
        self.log('Test/acc', acc, prog_bar=True, sync_dist=True)
        return {'loss': loss,
                'accuracy': acc}
    

    def test_epoch_end(self, outputs):
        self.test_acc.reset()
        return super().test_epoch_end(outputs)

@TASK.register_module()
class MultiLabelClassification (BaseDownstreamTask):

    def __init__(self, params: T.Dict[str, T.Any]) -> None:
        super().__init__(params)
        num_labels = getattr(self.head, 'num_classes', 1)
        self.criterion = nn.MultiLabelSoftMarginLoss()

        avg_method = params.get('avg_method', 'macro')
        self.mAP_map = {
            'train': tmc.MultilabelAveragePrecision(num_labels, avg_method),
            'val': tmc.MultilabelAveragePrecision(num_labels, avg_method),
            'test': tmc.MultilabelAveragePrecision(num_labels, avg_method)
        }
        self.training_method = params.get('training_method', PROBING)
        self.feature_opt = params.get('feature_opt', 'flatten')
        assert(self.training_method in TRAIN_METHODS)

    def forward(self, x: torch.Tensor):
        with torch.set_grad_enabled(self.training_method == FINETUNING):
            features = self.encoder(x)

        if self.feature_opt == 'flatten':
            features = features.flatten(1)
        elif self.feature_opt == 'cls_token':
            features = features[:, 0]

        x = self.head(features)
        return x

    def _shared_step(self, batch, split):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = torch.sigmoid(logit)
        self.mAP_map[split].update(pred, y)
        self.log(f'{split.capitalize()}/loss', loss, prog_bar=False,
                                               sync_dist=(split!="train"))
        return {'loss': loss}

    def on_train_epoch_start(self) -> None:
        if self.training_method == PROBING:
            self.encoder.eval()

    def training_step(self, batch, batch_nb):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_nb):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_nb):
        return self._shared_step(batch, 'test')

    def training_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Train/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)
        self.log('Train/epoch_avg_mAP', self.mAP_map['train'].compute(), logger=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Val/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)
        self.log('Val/epoch_avg_mAP', self.mAP_map['val'].compute(), logger=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Test/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)
        self.log('Test/epoch_avg_mAP', self.mAP_map['test'].compute(), logger=True, sync_dist=True)
