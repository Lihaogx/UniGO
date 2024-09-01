import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch.nn import Sequential, Linear, ReLU
from model.layer.odnet_layer import ODNETLayer
from torch_geometric.data import Data
from torchmetrics import Metric
from model.loss.dtw_loss import SoftDTW
import numpy as np
from model.loss.wasserstein_loss import torch_wasserstein_loss
import torch.nn.functional as F


# Dictionary to map layer types to corresponding PyTorch Geometric layer classes
def create_gin_layer(in_channels, out_channels):
    mlp = Sequential(Linear(in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
    return GINConv(nn=mlp, eps=0.1, train_eps=True)

GNN_layer = {'gcn': GCNConv,
         'gat': GATConv,
         'gin': create_gin_layer,
         'odnet': ODNETLayer
}

class SingleOrderGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout,loss_type, num_layers=2, order=1, layer_type='gcn', force_layers=None, **kargs):
        super(SingleOrderGNN, self).__init__()
        self.initial_layer = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        # Determine the number of layers
        if force_layers is not None:
            num_layers = force_layers
        else:
            num_layers = num_layers * order
        self.loss_type = loss_type
        # Create a list of layers
        self.layers = nn.ModuleList([
            GNN_layer[layer_type](hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])
        if 'cross_entropy' in self.loss_type:
            self.pred_layer = self.pred_layer = nn.Sequential(
                nn.Linear(hidden_channels, self.num_class),
            )
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, batch, edge_weight=None):
        x, edge_index = batch.x, batch.edge_index
        x = self.initial_layer(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = torch.relu(x)
            out = self.dropout(x)
        batch.pred = self.pred_layer(x).view(batch.batch_size, -1)
        batch.y = batch.y.view(batch.batch_size, -1)
        return batch
    
    def loss(self, batch, loss_type='mse'):
        if loss_type == 'wasserstein':
            predictions = batch.pred
            target = batch.y
            distance = torch_wasserstein_loss(target, predictions).mean()
            return distance
        elif loss_type == 'mse':
            predictions = batch.pred.flatten()
            target = batch.y.flatten()
            mse_loss = nn.MSELoss()
            loss = mse_loss(predictions, target)
            return loss
        elif loss_type == 'dtw':
            target = batch.y
            predictions = batch.pred
            # target, num_splits = reshape_for_dtw(target)
            # predictions, _ = reshape_for_dtw(predictions)
            dtw_loss = SoftDTW(self.gamma)
            loss = dtw_loss(target, predictions)
            # loss = compute_batch_loss(loss, num_splits)
            return loss
        elif loss_type == 'cross_entropy':
            logits = batch.pred.view(-1, self.num_class)
            step = 1 / self.num_class
            bins =  torch.arange(0, 1 + step, step, device=logits.device)
            target_labels = torch.bucketize(batch.y, bins, right=True).view(-1).long() - 1
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(logits, target_labels)
            return loss
        elif loss_type == 'cross_entropy+wd':
            
            batch_size, total_nodes = batch.pred.size(0), batch.pred.size(1)
            preds_softmax = F.softmax(batch.pred.view(batch_size, int(total_nodes/ self.num_class), self.num_class), dim=-1)
            preds_distr = preds_softmax.mean(dim=1)
            target_one_hot = F.one_hot(batch.y.long(), num_classes=self.num_class).to(torch.float)
            target_distr = target_one_hot.mean(dim=1)
            distance = torch_wasserstein_loss(target_distr, preds_distr)
            
            
            logits = batch.pred.view(-1, self.num_class)
            step = 1 / self.num_class
            bins =  torch.arange(0, 1 + step, step, device=logits.device)
            target_labels = torch.bucketize(batch.y, bins, right=True).view(-1).long() - 1
            ce_loss = nn.CrossEntropyLoss()
            loss_1 = ce_loss(logits, target_labels)
            
            loss = loss_1 +  self.m * distance
            return loss
