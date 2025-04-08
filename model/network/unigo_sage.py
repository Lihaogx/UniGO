import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling, EdgePooling, ASAPooling, PANPooling, MemPooling, SAGEConv
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
    def __init__(self, lookback, horizon, feature_dim, hid_dim, dropout):
        super(Refiner, self).__init__()
        self.feature_dim = feature_dim
        self.lookback = lookback
        self.horizon = horizon
        self.dropout = dropout

        self.mlp_X = nn.Sequential(
            nn.Linear(lookback * feature_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon * feature_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, horizon * feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, X, Y):


        X = X.permute(1, 0, 2).reshape(-1, self.lookback * self.feature_dim)  # [cluster_nodes, lookback * feature_dim]
        

        Y = Y.permute(1, 0, 2).reshape(-1, self.horizon * self.feature_dim)  # [cluster_nodes, horizon * feature_dim]
        

        X = self.mlp_X(X)  # [cluster_nodes, hid_dim]
        Y = self.mlp_Y(Y)  # [cluster_nodes, hid_dim]

        output = torch.cat([X, Y], dim=-1)  # [cluster_nodes, hid_dim * 2]
        

        refined_Y = self.mlp_out(output)  # [cluster_nodes, horizon * feature_dim]
        
        refined_Y = refined_Y.reshape(-1, self.horizon, self.feature_dim).permute(1, 0, 2)  # [horizon, cluster_nodes, feature_dim]
        
        return refined_Y

class GraphSAGE(nn.Module):
    def __init__(self, feature_dim, ode_hid_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(feature_dim, ode_hid_dim)
        self.conv2 = SAGEConv(ode_hid_dim, feature_dim)
        
        self.adj = None

    def forward(self, x):
        edge_index = self.adj.nonzero().t().contiguous()
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
class BackboneODE(nn.Module):
    """ dX/dt = f(X) + g(X, A)"""
    def __init__(self, lookback, feature_dim, ode_hid_dim, method, dropout):
        super(BackboneODE, self).__init__()
        
        self.method = method
        self.feature_dim = feature_dim
        

        self.init_enc = nn.Sequential(
            nn.Linear(lookback, ode_hid_dim, bias=True), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ode_hid_dim, 1, bias=True)
        )
        

        self.f = nn.Sequential(
            nn.Linear(feature_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ode_hid_dim, feature_dim, bias=True),
        )
        

        self.g = GraphSAGE(feature_dim, ode_hid_dim)
        
    def dxdt(self, t, x):
        """
        Right-hand side function of the differential equation dX/dt = f(X) + g(X, A)

        Args:
            t: Current time point (not used, but required by ODE solver)
            x: Current state, shape [num_supernodes, feature_dim]
            adj_w: Adjacency matrix, shape [num_supernodes, num_supernodes]
        
        Returns:
            dX/dt, shape [num_supernodes, feature_dim]
        """
        x_self = self.f(x)         # [num_supernodes, feature_dim]
        
        x_neigh = self.g(x) # [num_supernodes, feature_dim]
        dxdt = x_self + x_neigh    # [num_supernodes, feature_dim]
        dxdt = torch.clamp(dxdt, min=-1e3, max=1e3)
        if torch.isnan(dxdt).any() or torch.isinf(dxdt).any():
            print(f"NaN or Inf detected in dxdt at t={t}")
            print(f"x_self stats: min={x_self.min().item():.4f}, max={x_self.max().item():.4f}")
            print(f"x_neigh stats: min={x_neigh.min().item():.4f}, max={x_neigh.max().item():.4f}")
        return dxdt

    def forward(self, tspan, x, adj_w):
        """
        Forward propagation method

        Args:
            tspan: Time span for ODE solver, shape [horizon]
            x: Input tensor, shape [lookback, num_supernodes, ag_hid_dim]
            adj_w: Adjacency matrix, shape [num_supernodes, num_supernodes]
        
        Returns:
            out: Output after ODE solving, shape [horizon, num_supernodes, feature_dim]
        """
        
        # 1. Transform the dimensions of the input tensor
        # Input x has shape [lookback, num_supernodes, feature_dim]
        # Need to transpose to [num_supernodes, feature_dim, lookback] for linear layer processing
        tspan = tspan.to(x.device)
        x = x.permute(1, 2, 0)  # [num_supernodes, feature_dim, lookback]
        
        # 2. Use init_enc for encoding, mapping the lookback dimension to 1
        x = self.init_enc(x)  # [num_supernodes, feature_dim, 1]
        
        # 3. Remove the last dimension to get the initial state x0
        x0 = x.squeeze(-1)  # [num_supernodes, feature_dim]
        self.g.adj = adj_w
        # 4. ODE solving to compute future states
        # odeint takes function dxdt, initial state x0, time span tspan
        # dxdt needs to accept t, x, adj_w
        out = ode.odeint(self.dxdt, x0, tspan, method=self.method)  # [horizon, num_supernodes, feature_dim]
        # print("BackboneODE output stats:")
        # print(f"out: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        return out  # [horizon, num_supernodes, feature_dim]



class UniGONet_Sage(nn.Module):
    def __init__(self, args):
        super(UniGONet_Sage, self).__init__()
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
        start_t = (self.lookback) * self.dt
        end_t = (self.lookback + self.horizon - 1) * self.dt
        self.tspan = torch.linspace(start_t, end_t, self.horizon)
        if self.pool_type == 'sag':
            self.pool = SAGPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'topk':
            self.pool = TopKPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'asa':
            self.pool = ASAPooling(self.lookback, ratio=self.pool_ratio)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")


        self.repr_net_x = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ag_hid_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.ag_hid_dim),
        )
        self.repr_net_super = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ag_hid_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.ag_hid_dim),
        )
        self.softmax = nn.Softmax(dim=-1)


        self.agc_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ag_hid_dim, self.feature_dim),
        )
        self.tanh = nn.Tanh()

        self.BackboneODE = BackboneODE(self.lookback, self.feature_dim, self.ode_hid_dim, self.method, self.dropout)

        self.refiners = nn.ModuleList([
            Refiner(self.lookback, self.horizon, self.feature_dim, self.sr_hid_dim, self.dropout) 
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

        x = batch.x  # [num_nodes, lookback]
        edge_index = batch.edge_index  # [2, num_edges]
        cluster_node_indices = batch.cluster_node_indices  # [total_clusters_nodes]
        cluster_ptr = batch.cluster_ptr  # [num_clusters + 1]

        num_nodes = x.size(0)  # 节点数量
        lookback = x.size(1)  # 时间步数
        feature_dim = self.feature_dim  # lookback


        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + adj.t()  # 确保对称性（无向图）

        norm_lap = normalized_laplacian(adj)  # [num_nodes, num_nodes]


        pooled_x = self.pool(x, edge_index)[0]  # pooled_x: [num_supernodes, lookback]
        num_supernodes = pooled_x.size(0)

        x = x.permute(1, 0).unsqueeze(-1)  # [lookback, num_nodes, feature_dim]
        pooled_x = pooled_x.permute(1, 0).unsqueeze(-1)  # [lookback, num_supernodes, feature_dim]
        node_repr = self.repr_net_x(x)  # [lookback, num_nodes, ag_hid_dim]
        supernode_repr = self.repr_net_super(pooled_x)  # [lookback, num_supernodes, ag_hid_dim]

        node_repr_flat = node_repr.reshape(num_nodes, lookback * self.ag_hid_dim)          # [num_nodes, lookback * ag_hid_dim]
        supernode_repr_flat = supernode_repr.reshape(num_supernodes, lookback * self.ag_hid_dim)  # [num_supernodes, lookback * ag_hid_dim]
        similarity = torch.matmul(node_repr_flat, supernode_repr_flat.t())  # [num_nodes, num_supernodes]
        assignment_matrix = F.softmax(similarity, dim=1)  # [num_nodes, num_supernodes]


        adj_mean = adj.mean(dim=0, keepdim=True).expand_as(adj)  # [num_nodes, num_nodes]
        backbone = torch.matmul(assignment_matrix.t(), torch.matmul(adj_mean, assignment_matrix))  # [num_supernodes, num_supernodes]


        agc_repr = self.tanh(
            self.agc_mlp(
                torch.matmul(norm_lap, x)
            )
        )  # [lookback, num_nodes, feature_dim]

        supernode_embeddings = torch.matmul(assignment_matrix.t(), agc_repr)  # [lookback, num_supernodes, feature_dim]


        Y_supernode = self.BackboneODE(self.tspan, supernode_embeddings, backbone)  # [horizon, num_supernodes, feature_dim]


        Y_coarse = torch.matmul(Y_supernode.squeeze(-1), assignment_matrix.t()).unsqueeze(-1)  # [horizon, num_nodes, feature_dim]

        if self.refine:
            Y_refine = torch.zeros_like(Y_coarse)  # [horizon, num_nodes, feature_dim]
            if isolate:
                Y_coarse = Y_coarse.detach()


            num_clusters = len(torch.unique_consecutive(cluster_ptr)) - 1  # len(self.refiners)
            
            for k in range(num_clusters):
                start = cluster_ptr[k]
                end = cluster_ptr[k + 1]
                cluster_nodes = cluster_node_indices[start:end]  # 获取簇 k 的节点索引

                if cluster_nodes.numel() == 0:
                    continue
                else:
                    cluster_X = x[:, cluster_nodes, :]  # [lookback, cluster_nodes, feature_dim]
                    cluster_Y_coarse = Y_coarse[:, cluster_nodes, :]  # [horizon, cluster_nodes, feature_dim]

                    refined_output = self.refiners[k](cluster_X, cluster_Y_coarse)  # [horizon, cluster_nodes, feature_dim]

                    Y_refine[:, cluster_nodes, :] = refined_output
        else:
            Y_refine = Y_coarse
        if self.other_loss:
            return Y_refine.permute(1, 0, 2).squeeze(-1), batch.y , assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
        else:
            return Y_refine.permute(1, 0, 2).squeeze(-1), batch.y # , assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
    
    
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