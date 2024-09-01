import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from model.layer.odnet_layer import ODNETLayer
from torch_geometric.data import Data


def create_gin_layer(in_channels, out_channels):
    mlp = Sequential(Linear(in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
    return GINConv(nn=mlp, eps=0.1, train_eps=True)

GNN_layer = {'gcn': GCNConv,
         'gat': GATConv,
         'gin': create_gin_layer,
         'odnet': ODNETLayer
}

class MultiOrderGraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, order=2, layer_type='gcn', combine_mode='mean'):
        super(MultiOrderGraphLayer, self).__init__()
        self.order = order
        self.activation = ReLU()
        self.dropout = nn.Dropout(dropout)  
        self.layers = nn.ModuleList([
            GNN_layer[layer_type](in_channels, out_channels)
            for i in range(order)
        ])
        self.combine_mode = combine_mode  
        self.ln = nn.LayerNorm(normalized_shape=out_channels)
        if self.combine_mode == 'concat':
            self.combine = nn.Linear(out_channels * self.order, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        outputs = []
        for i, layer in enumerate(self.layers):
            out = layer(x, edge_index, edge_weight)
            out = self.activation(out)  
            out = self.dropout(out)  
            outputs.append(out)

        # Combine outputs from all orders
        if self.combine_mode == 'mean':
            return torch.mean(torch.stack(outputs), dim=0)
        elif self.combine_mode == 'add':
            return torch.sum(torch.stack(outputs), dim=0)
        elif self.combine_mode == 'concat':
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.combine(outputs)
            return outputs  
        else:
            raise ValueError("Invalid combine mode specified: choose 'mean', 'add', or 'concat'") 