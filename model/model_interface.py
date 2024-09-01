# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

Network = { 'mogo': MultiOrderGNN, 
           'single': SingleOrderGNN
}

class MInterface(pl.LightningModule):
    def __init__(self, model_config, optim_config):
        super().__init__()
        self.in_channels = model_config.in_dim
        self.hidden_channels = model_config.hidden_dim
        self.num_layers = model_config.layers
        self.order = model_config.order
        self.network_type = model_config.network_type
        self.layer_type = model_config.layer_type
        self.optim = optim_config
        self.loss_type = model_config.loss_type
        self.combine_mode = model_config.combine_mode
        self.force_layer = model_config.force_layer
        self.dropout = model_config.dropout
        self.gamma = model_config.gamma
        self.num_class = model_config.num_class
        self.loss_type = model_config.loss_type
        self.num_node = model_config.num_node
        self.cluster_type = model_config.cluster_type
        self.m = model_config.m
        self.val_metric = ODMetrics(num_calss=self.num_class, loss_type=self.loss_type, gamma=self.gamma)
        self.test_metric = ODMetrics(num_calss=self.num_class, loss_type=self.loss_type, gamma=self.gamma)

        self.load_model()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = self(batch)
        loss = self.model.loss(batch, self.loss_type)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self(batch)
        loss = self.model.loss(batch, self.loss_type)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_metric.update(batch.pred, batch.y)
        
        return loss

    def test_step(self, batch, batch_idx):
        batch = self(batch)
        loss = self.model.loss(batch, self.loss_type)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_metric.update(batch.pred, batch.y)
        
        return loss

    def on_validation_epoch_end(self):
        results = self.val_metric.compute()
        self.log_dict({'val_wasserstein_distance': results['wasserstein_distance'], 
                       'val_mse': results['mse'], 
                       'val_accuracy': results['accuracy'],
                       'val_f1':results['f1_score'],
                       'val_dtw':results['dtw']
                        }, prog_bar=True)
        self.val_metric.reset()
        # Make the Progress Bar leave there
        self.print('')

    def on_test_epoch_end(self):
        results = self.test_metric.compute()
        self.log_dict({'test_wasserstein_distance': results['wasserstein_distance'], 
                       'test_mse': results['mse'], 
                       'test_accuracy': results['accuracy'],
                       'test_f1':results['f1_score'],
                       'test_dtw':results['dtw']
                       }, prog_bar=True)
        self.test_metric.reset()
        # Make the Progress Bar leave there
        self.print('')
        
    def configure_optimizers(self):
        optimizer = create_optimizer(self.model.parameters(), self.optim)
        scheduler = create_scheduler(optimizer, self.optim)
        return [optimizer], [scheduler]

    def load_model(self):
        self.model = Network[self.network_type](in_channels = self.in_channels, 
                                                hidden_channels = self.hidden_channels, 
                                                m = self.m,
                                                cluster_type = self.cluster_type,
                                                num_layers = self.num_layers, 
                                                dropout = self.dropout,
                                                gamma = self.gamma,
                                                num_class = self.num_class,
                                                num_node = self.num_node,
                                                loss_type = self.loss_type,
                                                order = self.order, 
                                                layer_type = self.layer_type, 
                                                force_layer = self.force_layer,
                                                combine_mode = self.combine_mode)