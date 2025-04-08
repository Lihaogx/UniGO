import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling, EdgePooling, ASAPooling, PANPooling, MemPooling,SAGEConv
import torchdiffeq as ode



def sort_edge_index(edge_index):
    if edge_index.dim() == 1:
        edge_index = edge_index.view(1, -1).repeat(2, 1)
    elif edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should have shape [2, num_edges], but got {edge_index.shape}")
    
    _, perm = edge_index[1].sort()
    return edge_index[:, perm]

class Refiner(nn.Module):
    def __init__(self, lookback, horizon, hid_dim, dropout):
        super(Refiner, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.dropout = dropout
        self.mlp_X = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, horizon),
            nn.Sigmoid()
        )
        
    def forward(self, X, Y):
        """

        Args:
            X: Input tensor with shape [lookback, cluster_nodes, feature_dim]
            Y: Input tensor with shape [horizon, cluster_nodes, feature_dim]
        
        Returns:
            refined_Y: Refined predictions with shape [horizon, cluster_nodes, feature_dim]
        """
        X = self.mlp_X(X)  # [cluster_nodes, hid_dim]
        Y = self.mlp_Y(Y)  # [cluster_nodes, hid_dim]
        
        output = torch.cat([X, Y], dim=-1)  # [cluster_nodes, hid_dim * 2]
        
        refined_Y = self.mlp_out(output)  # [cluster_nodes, horizon]
        
        return refined_Y #  [horizon, cluster_nodes]


class GraphSAGEBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GraphSAGEBackbone, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.layers.append(SAGEConv(hidden_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        
        if x.shape[0] == 0 or edge_index.shape[1] == 0:
            raise ValueError("Empty input tensor or edge_index")
        edge_index = sort_edge_index(edge_index)
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        return x


class UniGONet_ReduceV2(nn.Module):
    def __init__(self, args):
        super(UniGONet_ReduceV2, self).__init__()
        self.args = args
        self.model_args = self.args.model
        self.feature_dim = self.model_args.feature_dim
        self.lookback = self.model_args.lookback
        self.horizon = self.model_args.horizon
        self.pool_ratio = self.model_args.pool_ratio
        self.sr_hid_dim = self.model_args.sr_hid_dim
        self.ag_hid_dim = self.model_args.ag_hid_dim
        self.ode_hid_dim = self.model_args.ode_hid_dim
        self.pool_type = self.model_args.pool_type
        self.method = self.model_args.method
        
        self.k = self.model_args.k * self.args.data.batch_size  # 假设 'k' 在 model_args 中定义
        self.dt = self.model_args.dt
        self.refine = self.model_args.refine
        self.other_loss = self.model_args.other_loss
        self.rg = self.model_args.rg
        self.onehot = self.model_args.onehot
        self.uniform = self.model_args.uniform
        self.refine_loss = self.model_args.refine_loss
        self.dropout = self.model_args.dropout
        self.kernel_size = self.model_args.kernel_size
        self.num_layers = self.model_args.num_layers
        start_t = (self.lookback) * self.dt
        end_t = (self.lookback + self.horizon - 1) * self.dt
        self.tspan = torch.linspace(start_t, end_t, self.horizon)
        # 定义池化层，输入维度为 lookback
        if self.pool_type == 'sag':
            self.pool = SAGPooling(self.sr_hid_dim, ratio=self.pool_ratio)
        elif self.pool_type == 'topk':
            self.pool = TopKPooling(self.sr_hid_dim, ratio=self.pool_ratio)
        elif self.pool_type == 'asa':
            self.pool = ASAPooling(self.sr_hid_dim, ratio=self.pool_ratio)
        elif self.pool_type == 'pan':
            self.pool = PANPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'mem':
            self.pool = MemPooling(
                        in_channels=self.lookback,  # 假设 lookback 对应输入通道数
                        out_channels=self.lookback,  # 输出通道数，可以根据需要调整
                        heads=1,  # 头的数量，可以根据需要调整
                        num_clusters=int(self.lookback * self.pool_ratio),  # 使用 pool_ratio 来确定聚类数
                        tau=1.0  # 温度参数，可以根据需要调整
                    )
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")

        # 表征网络
        self.repr_net_x = nn.Sequential(
            nn.Linear(self.lookback, self.ag_hid_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.ag_hid_dim),
        )
        self.repr_net_super = nn.Sequential(
            nn.Linear(self.lookback, self.ag_hid_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.ag_hid_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

        # 状态聚合网络
        self.agc_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ag_hid_dim, self.feature_dim),
        )
        self.tanh = nn.Tanh()

        self.graphsage = SAGEConv(self.ag_hid_dim, self.ag_hid_dim)

        self.Backbone = GraphSAGEBackbone(self.ode_hid_dim, self.ode_hid_dim, self.horizon, self.num_layers)

        self.refiners = nn.ModuleList([
            Refiner(self.lookback, self.horizon, self.sr_hid_dim, self.dropout) 
            for _ in range(self.k)
        ])
        
        self.shaped_refiner = Refiner(self.lookback, self.horizon, self.sr_hid_dim, self.dropout)

        self.mlp_refine = nn.Sequential(
            nn.Linear(self.horizon, self.sr_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.sr_hid_dim, self.horizon),
            nn.Sigmoid()
        )

    def forward(self, batch, isolate=False):
        """
        Forward pass of MogoNet.

        Args:
            batch: A batch object containing graph data.
            tspan: Time span for the ODE solver.
            isolate: If True, detaches Y_coarse from the computation graph.

        Returns:
            assignment_matrix: Soft assignment matrix [num_nodes, num_supernodes].
            Y_refine: Refined predictions [batch_size, horizon, num_nodes, feature_dim].
            Y_supernode: Supernode predictions [batch_size, horizon, num_supernodes, feature_dim].
            additional_outputs: Tuple containing (Y_coarse, x, supernode_embeddings).
        """
        x = batch.x  # [num_nodes, lookback]
        edge_index = batch.edge_index  # [2, num_edges]
        cluster_node_indices = batch.cluster_node_indices  # [total_clusters_nodes]
        cluster_ptr = batch.cluster_ptr  # [num_clusters + 1]

        num_nodes = x.size(0)  # 
        lookback = x.size(1)  # 
        feature_dim = self.feature_dim  # lookback


        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + adj.t()  
        # adj.fill_diagonal_(1)  

        node_repr = self.repr_net_x(x)  # [num_nodes, ag_hid_dim]
        agc_repr = self.graphsage(node_repr, edge_index)
        agc_repr = self.tanh(agc_repr)

        supernode_repr, pooled_edge_index, pooled_edge_attr, pooled_batch, perm, score = self.pool(agc_repr, edge_index, edge_attr=None, batch=batch.batch)  # pooled_x: [num_supernodes, lookback]

        

        assignment_matrix = self.assignment_matrix(agc_repr, supernode_repr, batch.batch, pooled_batch)


        temp = torch.matmul(adj, assignment_matrix)  # [num_nodes, num_supernodes]
        backbone = torch.matmul(assignment_matrix.transpose(0, 1), temp)  # [num_supernodes, num_supernodes]



        backbone_edge_index = (backbone > 0).nonzero().t()
        Y_supernode = self.Backbone(supernode_repr, backbone_edge_index)  # [horizon, num_supernodes]


        Y_coarse = torch.matmul(assignment_matrix, Y_supernode)  # [horizon, num_nodes]

        if self.refine:
            if self.refine == 'shape':
                Y_refine = self.shaped_refiner(agc_repr, Y_coarse)
            else:
                Y_refine = torch.zeros_like(Y_coarse)  # [horizon, num_nodes]
                if isolate:
                    Y_coarse = Y_coarse.detach()


                num_clusters = len(torch.unique_consecutive(cluster_ptr)) - 1  # len(self.refiners)
                
                for k in range(num_clusters):
                    start = cluster_ptr[k]
                    end = cluster_ptr[k + 1]
                    cluster_nodes = cluster_node_indices[start:end]  

                    if cluster_nodes.numel() == 0:
                        continue
                    else:
                        cluster_X = agc_repr[cluster_nodes, :]  # [cluster_nodes, lookback]
                        cluster_Y_coarse = Y_coarse[cluster_nodes, :]  # [cluster_nodes, horizon]

                        refined_output = self.refiners[k](cluster_X, cluster_Y_coarse)  # [horizon, cluster_nodes]

                        Y_refine[cluster_nodes, : ] = refined_output
        else:
            Y_refine = Y_coarse
            
            
        Y_refine = self.mlp_refine(Y_refine)
        if self.other_loss:
            return Y_refine, batch.y , assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
        else:
            return Y_refine, batch.y # assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
        
            
    def assignment_matrix(self, node_repr, supernode_repr, batch, pooled_batch):
        """
        Build a global assignment matrix that maps original nodes to pooled nodes (supernodes).
        
        Args:
            node_repr: Original node representations with shape [num_nodes, feature_dim]
            supernode_repr: Pooled node representations with shape [num_pool_nodes, feature_dim]
            batch: Original node batch vector with shape [num_nodes]
            pooled_batch: Pooled node batch vector with shape [num_pool_nodes]
        
        Returns:
            assignment_matrix: Dense assignment matrix with shape [num_nodes, num_pool_nodes]
        """
        device = node_repr.device
        

        unique_batches = batch.unique()


        batch_sorted_indices = batch.argsort()
        pool_batch_sorted_indices = pooled_batch.argsort()

        node_repr = node_repr[batch_sorted_indices]
        node_batch = batch[batch_sorted_indices]

        supernode_repr = supernode_repr[pool_batch_sorted_indices]
        supernode_batch = pooled_batch[pool_batch_sorted_indices]

        total_num_nodes = node_repr.size(0)
        total_num_supernodes = supernode_repr.size(0)



        assignment_matrix = torch.zeros((total_num_nodes, total_num_supernodes), device=device)

        start_node = 0
        start_supernode = 0

        for graph_id in unique_batches:
            
            node_mask = (node_batch == graph_id)
            supernode_mask = (supernode_batch == graph_id)

            node_repr_sub = node_repr[node_mask]
            supernode_repr_sub = supernode_repr[supernode_mask]

            num_nodes = node_repr_sub.shape[0]
            num_supernodes = supernode_repr_sub.shape[0]

            if num_supernodes == 0:
                start_node += num_nodes
                continue

            similarity_sub = torch.matmul(node_repr_sub, supernode_repr_sub.t())

            assignment_matrix_sub = F.softmax(similarity_sub, dim=1)


            assignment_matrix[start_node:start_node+num_nodes, start_supernode:start_supernode+num_supernodes] = assignment_matrix_sub


            assigned_submatrix = assignment_matrix[start_node:start_node+num_nodes, start_supernode:start_supernode+num_supernodes]

            start_node += num_nodes
            start_supernode += num_supernodes

        return assignment_matrix
        
    def loss(self, pred, target, *args):

        pred_loss = F.mse_loss(pred, target)


        total_loss = pred_loss

        if self.other_loss and len(args) >= 3:
            assignment_matrix, supernode_adj, orig_adj = args[:3]
            

            rg_loss, _ = self._rg_loss(args[3], target, assignment_matrix) if len(args) > 3 else (0, None)


            onehot_loss = self._onehot_loss(assignment_matrix)

            uniform_loss = self._uniform_loss(assignment_matrix)


            # recons_loss = self._recons_loss(assignment_matrix, orig_adj)
            if self.refine:

                refine_loss, _ = self._refine_loss(args[4], target) if len(args) > 4 else (0, None)
            else:
                refine_loss = 0
            total_loss += (
                self.rg * rg_loss +
                self.onehot * onehot_loss +
                self.uniform * uniform_loss +
                self.refine_loss * refine_loss
            )

        return total_loss

    def _agc_state(self, X, assignment_matrix):
        agc_repr = self.tanh(self.agc_mlp(self.norm_lap @ X))
        X_supernode = assignment_matrix @ agc_repr # lookback, supernode_num, feature_dim
        return X_supernode

    def _rg_loss(self, y_rg, Y, assignment_matrix, dim=None):
        with torch.no_grad():
            Y_supernode = self._agc_state(Y, assignment_matrix)
        if dim is None:
            rg_loss = torch.mean((y_rg - Y_supernode) ** 2)
        else:
            rg_loss = torch.mean((y_rg - Y_supernode) ** 2, dim=dim)
        return rg_loss, Y_supernode

    def _onehot_loss(self, assignment_matrix):
        entropy = -torch.sum(assignment_matrix * torch.log2(assignment_matrix + 1e-5), dim=0)
        onehot_loss = torch.mean(entropy)
        return onehot_loss

    def _uniform_loss(self, assignment_matrix):
        supernode_strength = torch.sum(assignment_matrix, dim=1)
        prob = supernode_strength / torch.sum(supernode_strength)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-5), dim=0)
        uniform_loss = -torch.mean(entropy) # maximize entropy
        return uniform_loss

    def _recons_loss(self, assignment_matrix, adj):
        surrogate_adj = assignment_matrix.T @ assignment_matrix
        recons_loss = torch.norm(adj - surrogate_adj, p='fro')
        return recons_loss

    def _refine_loss(self, y_refine, Y, dim=None):
        if dim is None:
            refine_loss = torch.mean((y_refine - Y) ** 2)
        else:
            refine_loss = torch.mean((y_refine - Y) ** 2, dim=dim)
        return refine_loss, Y
    
    