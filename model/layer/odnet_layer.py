import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class ODNETLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ODNETLayer, self).__init__(aggr='add')  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.phi = nn.Linear(in_channels * 2, out_channels)
        self.psi = nn.Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.phi.reset_parameters()
        self.psi.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform node feature matrix with psi
        self_loop_attr = self.psi(x)
        
        # Propagate messages using the phi transformation
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, self_loop_attr=self_loop_attr)
    
    def message(self, x_i, x_j, self_loop_attr_i, edge_index, size):
        # x_i is [E, in_channels]
        # x_j is [E, in_channels]
        
        tmp = torch.cat([x_i, x_j - x_i], dim=-1)
        
        return self.phi(tmp)
    
    def update(self, aggr_out, self_loop_attr):
        return aggr_out + self_loop_attr