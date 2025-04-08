import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling, EdgePooling, ASAPooling, PANPooling, GCNConv, GATConv, SAGEConv, GraphConv, GINConv, SGConv
import torchdiffeq as ode

def normalized_laplacian(A: torch.Tensor):
    """Symmetrically Normalized Laplacian: I - D^-1/2 * ( A ) * D^-1/2"""
    out_degree = torch.sum(A, dim=1)
    int_degree = torch.sum(A, dim=0)
    
    out_degree_sqrt_inv = torch.pow(out_degree, -0.5)
    int_degree_sqrt_inv = torch.pow(int_degree, -0.5)
    mx_operator = torch.eye(A.shape[0], device=A.device) - torch.diag(out_degree_sqrt_inv) @ A @ torch.diag(int_degree_sqrt_inv)
    
    return mx_operator


class Refiner(nn.Module):
    def __init__(self, lookback, horizon, feature_dim, hid_dim):
        super(Refiner, self).__init__()
        self.feature_dim = feature_dim
        self.lookback = lookback
        self.horizon = horizon
        
        # MLP for processing X
        self.mlp_X = nn.Sequential(
            nn.Linear(lookback * feature_dim, hid_dim),
            nn.Tanh(),
        )
        # MLP for processing Y
        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon * feature_dim, hid_dim),
            nn.Tanh(),
        )
        # Output MLP
        self.mlp_out = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, horizon * feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, X, Y):
        """
        Forward pass method

        Args:
            X: Input tensor with shape [lookback, cluster_nodes, feature_dim]
            Y: Input tensor with shape [horizon, cluster_nodes, feature_dim]
        
        Returns:
            refined_Y: Refined predictions with shape [horizon, cluster_nodes, feature_dim]
        """
        # Transform X from [lookback, cluster_nodes, feature_dim] to [cluster_nodes, lookback * feature_dim]
        X = X.permute(1, 0, 2).reshape(-1, self.lookback * self.feature_dim)  # [cluster_nodes, lookback * feature_dim]
        
        # Transform Y from [horizon, cluster_nodes, feature_dim] to [cluster_nodes, horizon * feature_dim]
        Y = Y.permute(1, 0, 2).reshape(-1, self.horizon * self.feature_dim)  # [cluster_nodes, horizon * feature_dim]
        
        # Process X and Y through their respective MLPs
        X = self.mlp_X(X)  # [cluster_nodes, hid_dim]
        Y = self.mlp_Y(Y)  # [cluster_nodes, hid_dim]
        
        # Concatenate outputs of X and Y
        output = torch.cat([X, Y], dim=-1)  # [cluster_nodes, hid_dim * 2]
        
        # Generate refined Y through output MLP
        refined_Y = self.mlp_out(output)  # [cluster_nodes, horizon * feature_dim]
        
        # Transform refined_Y from [cluster_nodes, horizon * feature_dim] to [horizon, cluster_nodes, feature_dim]
        refined_Y = refined_Y.reshape(-1, self.horizon, self.feature_dim).permute(1, 0, 2)  # [horizon, cluster_nodes, feature_dim]
        
        return refined_Y

class BackboneGNN(nn.Module):
    def __init__(self, lookback, horizon, num_layers, gnn_type='GCN', feature_dim=1):
        super(BackboneGNN, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.feature_dim = feature_dim
        
        # GNN network
        self.gnn = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.gnn.append(self._get_gnn_layer(lookback * feature_dim, horizon, gnn_type))
        self.batch_norms.append(nn.BatchNorm1d(horizon))

        # Middle layers
        for _ in range(num_layers - 2):
            self.gnn.append(self._get_gnn_layer(horizon, horizon, gnn_type))
            self.batch_norms.append(nn.BatchNorm1d(horizon))

        # Last layer
        self.gnn.append(self._get_gnn_layer(horizon, horizon, gnn_type))

    def _get_gnn_layer(self, in_dim, out_dim, gnn_type):
        if gnn_type == 'GCN':
            return GCNConv(in_dim, out_dim)
        elif gnn_type == 'GIN':
            nn_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(nn_layer)
        elif gnn_type == 'SG':
            return SGConv(in_dim, out_dim, K=3)
        elif gnn_type == 'GAT':
            return GATConv(in_dim, out_dim)
        elif gnn_type == 'SAGE':
            return SAGEConv(in_dim, out_dim)
        elif gnn_type == 'GraphConv':
            return GraphConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x, edge_index):
        """
        Forward pass method

        Args:
            tspan: Time span for generating output sequence, shape [horizon]
            x: Input tensor with shape [lookback, num_supernodes, feature_dim]
            edge_index: Edge indices with shape [2, num_edges]
        
        Returns:
            out: GNN output sequence with shape [horizon, num_supernodes, feature_dim]
        """
        
        # 1. Transform input tensor dimensions
        x = x.permute(1, 0, 2)  # [num_supernodes, lookback, feature_dim]
        x = x.reshape(x.shape[0], -1)  # [num_supernodes, lookback * feature_dim]
        
        # 2. Generate output sequence using GNN
        for i, layer in enumerate(self.gnn):
            x = layer(x, edge_index)
            if i < len(self.gnn) - 1:  # Don't apply BatchNorm and ReLU to the last layer
                x = self.batch_norms[i](x)
                x = F.relu(x)
        
        # 3. Adjust output dimensions to match expected shape
        out = x.unsqueeze(-1)  # [num_supernodes, horizon, 1]
        out = out.permute(1, 0, 2)  # [horizon, num_supernodes, 1]
        
        return out



class UniGONet_GNN(nn.Module):
    def __init__(self, args):
        super(UniGONet_GNN, self).__init__()
        self.args = args
        self.model_args = self.args.model
        self.feature_dim = self.model_args.feature_dim
        self.lookback = self.model_args.lookback
        self.horizon = self.model_args.horizon
        self.pool_ratio = self.model_args.pool_ratio
        self.sr_hid_dim = self.model_args.sr_hid_dim
        self.ag_hid_dim = self.model_args.ag_hid_dim
        self.ode_hid_dim = self.model_args.ode_hid_dim
        self.gnn_type = self.model_args.gnn_type
        self.gnn_layers = self.model_args.num_layers
        self.pool_type = self.model_args.pool_type
        self.method = self.model_args.method
        self.k = self.model_args.k * self.args.data.batch_size  # Assuming 'k' is defined in model_args
        self.dt = self.model_args.dt
        
        start_t = (self.lookback) * self.dt
        end_t = (self.lookback + self.horizon - 1) * self.dt
        self.tspan = torch.linspace(start_t, end_t, self.horizon)
        
        # Define pooling layer with input dimension lookback
        if self.pool_type == 'sag':
            self.pool = SAGPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'topk':
            self.pool = TopKPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'asa':
            self.pool = ASAPooling(self.lookback, ratio=self.pool_ratio)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")

        # Representation network
        self.repr_net_x = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.Tanh(),
            nn.Linear(self.ag_hid_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.ag_hid_dim),
        )
        self.repr_net_super = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.Tanh(),
            nn.Linear(self.ag_hid_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.ag_hid_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

        # State aggregation network
        self.agc_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.Linear(self.ag_hid_dim, self.feature_dim),
        )
        self.tanh = nn.Tanh()

        # Backbone dynamics
        self.BackboneGNN = BackboneGNN(self.lookback, self.horizon, self.gnn_layers, self.gnn_type)

        # Refinement layers
        self.refiners = nn.ModuleList([
            Refiner(self.lookback, self.horizon, self.feature_dim, self.sr_hid_dim) 
            for _ in range(self.k)
        ])

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
        # Step 1: Read data without modifying dimensions
        x = batch.x  # [num_nodes, lookback]
        edge_index = batch.edge_index  # [2, num_edges]
        cluster_node_indices = batch.cluster_node_indices  # [total_clusters_nodes]
        cluster_ptr = batch.cluster_ptr  # [num_clusters + 1]

        num_nodes = x.size(0)  # Number of nodes
        lookback = x.size(1)  # Number of time steps
        feature_dim = self.feature_dim  # lookback

        # Step 2: Generate adjacency matrix adj from edge_index
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + adj.t()  # Ensure symmetry (undirected graph)

        # Calculate normalized Laplacian matrix
        norm_lap = normalized_laplacian(adj)  # [num_nodes, num_nodes]

        # Step 3: Pooling operation
        pooled_x = self.pool(x, edge_index)[0]  # pooled_x: [num_supernodes, lookback]
        num_supernodes = pooled_x.size(0)
        
        # Step 4: Representation network
        x = x.permute(1, 0).unsqueeze(-1)  # [lookback, num_nodes, feature_dim]
        pooled_x = pooled_x.permute(1, 0).unsqueeze(-1)  # [lookback, num_supernodes, feature_dim]
        node_repr = self.repr_net_x(x)  # [lookback, num_nodes, ag_hid_dim]
        supernode_repr = self.repr_net_super(pooled_x)  # [lookback, num_supernodes, ag_hid_dim]
        
        # Calculate similarity between node representations and supernode representations
        node_repr_flat = node_repr.reshape(num_nodes, lookback * self.ag_hid_dim)          # [num_nodes, lookback * ag_hid_dim]
        supernode_repr_flat = supernode_repr.reshape(num_supernodes, lookback * self.ag_hid_dim)  # [num_supernodes, lookback * ag_hid_dim]
        similarity = torch.matmul(node_repr_flat, supernode_repr_flat.t())  # [num_nodes, num_supernodes]
        assignment_matrix = F.softmax(similarity, dim=1)  # [num_nodes, num_supernodes]

        # Calculate supernode adjacency matrix
        adj_mean = adj.mean(dim=0, keepdim=True).expand_as(adj)  # [num_nodes, num_nodes]
        backbone = torch.matmul(assignment_matrix.t(), torch.matmul(adj_mean, assignment_matrix))  # [num_supernodes, num_supernodes]

        # Step 5: State aggregation
        agc_repr = self.tanh(
            self.agc_mlp(
                torch.matmul(norm_lap, x)
            )
        )  # [lookback, num_nodes, feature_dim]

        # Get supernode embeddings using assignment matrix
        supernode_embeddings = torch.matmul(assignment_matrix.t(), agc_repr)  # [lookback, num_supernodes, feature_dim]

        # Step 6: Backbone dynamics
        # Convert adjacency matrix to edge_index format
        edge_index = torch.nonzero(backbone).t().contiguous()
        Y_supernode = self.BackboneGNN(supernode_embeddings, edge_index)  # [horizon, num_supernodes, feature_dim]

        # Step 7: Map back to original nodes
        Y_coarse = torch.matmul(Y_supernode.squeeze(-1), assignment_matrix.t()).unsqueeze(-1)  # [horizon, num_nodes, feature_dim]

        Y_refine = torch.zeros_like(Y_coarse)  # [horizon, num_nodes, feature_dim]
        if isolate:
            Y_coarse = Y_coarse.detach()

        # Use Refiner to refine predictions
        num_clusters = len(torch.unique_consecutive(cluster_ptr)) - 1  # len(self.refiners)
        
        for k in range(num_clusters):
            start = cluster_ptr[k]
            end = cluster_ptr[k + 1]
            cluster_nodes = cluster_node_indices[start:end]  # Get node indices for cluster k

            if cluster_nodes.numel() == 0:
                continue
            else:
                cluster_X = x[:, cluster_nodes, :]  # [lookback, cluster_nodes, feature_dim]
                cluster_Y_coarse = Y_coarse[:, cluster_nodes, :]  # [horizon, cluster_nodes, feature_dim]
                # Use corresponding refiner to refine predictions for nodes in the cluster
                refined_output = self.refiners[k](cluster_X, cluster_Y_coarse)  # [horizon, cluster_nodes, feature_dim]
                # Write refined predictions back to Y_refine
                Y_refine[:, cluster_nodes, :] = refined_output
        
        return Y_refine.permute(1, 0, 2).squeeze(-1), batch.y # , assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
    
    def loss(self, pred, target):
        # Prediction loss (MSE)
        pred_loss = F.mse_loss(pred, target)
        
        # Total loss
        total_loss = pred_loss

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