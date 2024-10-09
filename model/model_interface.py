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
sys.path.append('/home/lh/UniGO/')
from utils import create_optimizer, create_scheduler
from model.network.gnn import FlexibleGNN
from model.network.unigo import UniGONet
from model.network.unigo_gnn import UniGONet_GNN
from model.network.unigo_sage import UniGONet_Sage
from model.network.unigo_reduce import UniGONet_Reduce
from model.metric.ODMetrics import ODMetrics
model = {'gnn':FlexibleGNN, 'unigo':UniGONet, 'unigo_gnn':UniGONet_GNN, 'unigo_sage':UniGONet_Sage, "unigo_reduce":UniGONet_Reduce}


class MInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.optim = self.args.optim
        self.val_metric = ODMetrics()
        self.test_metric = ODMetrics()

        self.load_model()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if self.args.model.other_loss:
            pred, target, *args = self(batch)
            loss = self.model.loss(pred, target, *args)
        else:
            pred, target = self(batch)
            loss = self.model.loss(pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.model.other_loss:
            pred, target, *args = self(batch)
            loss = self.model.loss(pred, target, *args)
        else:
            pred, target = self(batch)
            loss = self.model.loss(pred, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_metric.update(pred, target)
        return loss

    def test_step(self, batch, batch_idx):
        if self.args.model.other_loss:
            pred, target, *args = self(batch)
            loss = self.model.loss(pred, target, *args)
        else:
            pred, target = self(batch)
            loss = self.model.loss(pred, target)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_metric.update(pred, target)
        
        return loss

    def on_validation_epoch_end(self):
        results = self.val_metric.compute()
        if torch.isnan(results['wasserstein_distance']).any():
            raise ValueError(f"NaN detected in results['wasserstein_distance']")
        self.log_dict({'val_wasserstein_distance': results['wasserstein_distance'], 
                       'val_mse': results['mse'], 
                        }, prog_bar=True)
        self.val_metric.reset()
        # Make the Progress Bar leave there
        self.print('')

    def on_test_epoch_end(self):
        results = self.test_metric.compute()
        self.log_dict({'test_wasserstein_distance': results['wasserstein_distance'], 
                       'test_mse': results['mse'], 
                       }, prog_bar=True)
        self.test_metric.reset()
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model.parameters(), self.optim)
        scheduler = create_scheduler(optimizer, self.optim)
        return [optimizer], [scheduler]
    

    def load_model(self):
        self.model = model[self.args.model_type](self.args)