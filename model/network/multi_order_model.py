import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data
from torchmetrics import Metric
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from model.layer.multiorder_layer import MultiOrderGraphLayer 
from model.layer.soft_cluster_pooling import SoftClusterPooling
import sys
from model.loss.wasserstein_loss import torch_wasserstein_loss
from model.loss.dtw_loss import SoftDTW
import torch.nn.functional as F

class ODMetrics(Metric):
    def __init__(self, dist_sync_on_step=False, loss_type='mse',  num_calss=4, gamma=0.1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("distances", default=[], dist_reduce_fx="mean")
        self.add_state("mses", default=[], dist_reduce_fx="mean")
        self.add_state("accuracies", default=[], dist_reduce_fx="mean")
        self.add_state("f1_scores", default=[], dist_reduce_fx="mean")
        self.add_state("dtws", default=[], dist_reduce_fx="mean")
        self.gamma = gamma
        self.num_class = num_calss
        self.loss_type = loss_type
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = preds.view(-1), target.view(-1)
        if 'cross_entropy' in self.loss_type:
            batch_size, total_nodes = preds.size(0), preds.size(1)
            preds_softmax = F.softmax(preds.view(batch_size, int(total_nodes/ self.num_class), self.num_class), dim=-1)
            preds_distr = preds_softmax.mean(dim=1)
            target_one_hot = F.one_hot(target.long(), num_classes=self.num_class).to(torch.float)
            target_distr = target_one_hot.mean(dim=1)
            distance = torch_wasserstein_loss(target_distr, preds_distr)
            mse = torch.Tensor(1)
            dtw = torch.Tensor(1)
            
            logits = preds.view(-1, self.num_class)
            step = 1 / self.num_class
            bins =  torch.arange(0, 1 + step, step, device=logits.device)
            target_labels = torch.bucketize(target, bins, right=True).view(-1).long() - 1
            _, predicted_labels = torch.max(logits, dim=1)
            correct_predictions = (predicted_labels == target_labels).float()
            accuracy = correct_predictions.mean()
            

            f1 = f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
            
        else:
            distance = torch_wasserstein_loss(target, preds)
            
            mse = nn.functional.mse_loss(preds, target)
            
            dtw_loss = SoftDTW(self.gamma)
            dtw = dtw_loss(preds, target)
            step = 1 / self.num_class
            bins =  torch.arange(0, 1 + step, step, device=preds.device)
            predicted_labels = torch.bucketize(preds, bins, right=True).view(-1)
            target_labels = torch.bucketize(target, bins, right=True).view(-1)
            accuracy = torch.mean((predicted_labels == target_labels).float())
            
            f1 = f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
            
            
        if isinstance(distance, torch.Tensor):
            self.distances.append(distance.cpu())
        self.mses.append(mse)
        self.dtws.append(dtw)
        step = 1 / self.num_class

        self.accuracies.append(accuracy)
        
        self.f1_scores.append(torch.tensor(f1, device=predicted_labels.device))
        

        
    def compute(self):
        return {
            'wasserstein_distance': torch.mean(torch.stack(self.distances)),
            'mse': torch.mean(torch.stack(self.mses)),
            'accuracy': torch.mean(torch.stack(self.accuracies)),
            'f1_score': torch.mean(torch.stack(self.f1_scores)),
            'dtw': torch.mean(torch.stack(self.dtws))
        }



class MultiOrderGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, gamma, num_class, num_node, cluster_type, loss_type, m, num_layers=2, order=2, layer_type='gcn', combine_mode='mean', **kargs):
        super(MultiOrderGNN, self).__init__()
        self.num_node = num_node
        self.initial_layer = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MultiOrderGraphLayer(hidden_channels, hidden_channels, dropout, order, layer_type, combine_mode))
            # self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            if i < num_layers - 1:  
                self.pooling_layers.append(SoftClusterPooling(hidden_channels, self.num_node, cluster_type))
        self.gamma = gamma
        self.num_class = num_class

        self.loss_type = loss_type
        self.bn = nn.BatchNorm1d(num_features=hidden_channels)

        self.ln = nn.LayerNorm(normalized_shape=hidden_channels)
        self.m = m
        if 'cross_entropy' in self.loss_type:
            self.pred_layer = self.pred_layer = nn.Sequential(
                nn.Linear(hidden_channels, self.num_class),
            )
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
        # self.pred_layer = nn.Linear(hidden_channels, 1)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.initial_layer(x)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MultiOrderGraphLayer):
                x = layer(x, edge_index)
            else:
                x = layer(x)
            x = self.ln(x)
            x = F.relu(x)
            if i < len(self.pooling_layers):
                x, _ = self.pooling_layers[i](x)
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
        
        elif loss_type == 'mse+wd':
            target = batch.y
            predictions = batch.pred

            mse_loss = nn.MSELoss()
            loss_1 = mse_loss(predictions.flatten(), target.flatten())
            
            distance = torch_wasserstein_loss(target, predictions).mean()

            return 10 * loss_1 + distance
            