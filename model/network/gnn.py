import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv, SGConv

class FlexibleGNN(nn.Module):
    def __init__(self, args):
        super(FlexibleGNN, self).__init__()
        self.args = args
        self.model_args = args.model
        self.lookback = self.model_args.lookback
        self.horizon = self.model_args.horizon
        self.hidden_dim = self.model_args.ag_hid_dim
        self.num_layers = self.model_args.num_layers
        self.gnn_type = self.model_args.gnn_type

        # 输入层
        self.input_layer = nn.Linear(self.lookback, self.hidden_dim)

        # GNN层
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            
            if self.gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            elif self.gnn_type == 'GIN':
                nn_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
                self.gnn_layers.append(GINConv(nn_layer))
            elif self.gnn_type == 'SG':
                self.gnn_layers.append(SGConv(self.hidden_dim, self.hidden_dim, K=3))
            elif self.gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(self.hidden_dim, self.hidden_dim))
            elif self.gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(self.hidden_dim, self.hidden_dim))
            elif self.gnn_type == 'GraphConv':
                self.gnn_layers.append(GraphConv(self.hidden_dim, self.hidden_dim))
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

        # 输出层
        self.output_layer = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        # 输入层
        x = self.input_layer(x)

        # GNN层
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)

        # 输出层
        out = self.output_layer(x)

        return out, batch.y  # [num_nodes, horizon]

    def loss(self, pred, target):
        return F.mse_loss(pred, target)