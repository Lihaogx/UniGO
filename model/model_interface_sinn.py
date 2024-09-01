import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
import sys
sys.path.append('/home/lh/MOGO/')
from utils import create_optimizer, create_scheduler
from model.network.multi_order_model import MultiOrderGNN, ODMetrics
from model.network.single_order_model import SingleOrderGNN
from model.network.sinn_model import SINN, SINN_ODMetrics

Network = { 'mogo': MultiOrderGNN, 
           'single': SingleOrderGNN, 
           'sinn': SINN,
           'nn': SINN
}

class MInterface(pl.LightningModule):
    def __init__(self, model_config, optim_config):
        super().__init__()
        self.num_users = model_config.num_users
        self.hidden_channels = model_config.hidden_dim
        self.num_layers = model_config.layers
        self.nclasses = model_config.nclasses
        self.type_odm = model_config.type_odm
        self.alpha = model_config.alpha
        self.beta = model_config.beta
        self.K = model_config.K
        self.act_type = model_config.activation_func
        self.in_channels = model_config.in_dim
        self.network_type = model_config.network_type
        self.optim = optim_config
        self.val_metric = SINN_ODMetrics()
        self.test_metric = SINN_ODMetrics()
        self.load_model()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = self(batch)
        loss = self.model.loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            batch = self(batch)
            loss = self.model.loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_metric.update('val', loss = loss)
        
        return loss

    def test_step(self, batch, batch_idx):
        res = self.model.prediction(batch)
        self.test_metric.update('test', pred_labels = res[0]['pred_label'], target_labels = batch['ground_truth_opinion'])

    def on_validation_epoch_end(self):
        results = self.val_metric.compute('val')
        self.log_dict({'val_loss': results['loss'],
                        }, prog_bar=True)
        self.val_metric.reset()
        # Make the Progress Bar leave there
        self.print('')

    def on_test_epoch_end(self):
        results = self.test_metric.compute('test')
        self.log_dict({ 'test_accuracy': results['accuracy'],
                       'test_f1':results['f1_score'],
                       }, prog_bar=True)
        self.test_metric.reset()
        # Make the Progress Bar leave there
        self.print('')
        
    def configure_optimizers(self):
        optimizer = create_optimizer(self.model.parameters(), self.optim)
        scheduler = create_scheduler(optimizer, self.optim)
        return [optimizer], [scheduler]

    def load_model(self):
        self.model = Network[self.network_type](num_users = self.num_users, 
                                                network_type = self.network_type,
                                                act_type = self.act_type, 
                                                hidden_channels = self.hidden_channels, 
                                                num_layers = self.num_layers, 
                                                nclasses = self.nclasses,
                                                type_odm = self.type_odm,
                                                alpha = self.alpha, 
                                                beta = self.beta, 
                                                K = self.K,
                                                )