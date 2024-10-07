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
        
        # 处理 X 的 MLP
        self.mlp_X = nn.Sequential(
            nn.Linear(lookback * feature_dim, hid_dim),
            nn.Tanh(),
        )
        # 处理 Y 的 MLP
        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon * feature_dim, hid_dim),
            nn.Tanh(),
        )
        # 输出的 MLP
        self.mlp_out = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, horizon * feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, X, Y):
        """
        前向传播方法

        参数:
            X: 输入张量，形状为 [lookback, cluster_nodes, feature_dim]
            Y: 输入张量，形状为 [horizon, cluster_nodes, feature_dim]
        
        返回:
            refined_Y: 精炼后的预测，形状为 [horizon, cluster_nodes, feature_dim]
        """
        # 将 X 从 [lookback, cluster_nodes, feature_dim] 转换为 [cluster_nodes, lookback * feature_dim]
        X = X.permute(1, 0, 2).reshape(-1, self.lookback * self.feature_dim)  # [cluster_nodes, lookback * feature_dim]
        
        # 将 Y 从 [horizon, cluster_nodes, feature_dim] 转换为 [cluster_nodes, horizon * feature_dim]
        Y = Y.permute(1, 0, 2).reshape(-1, self.horizon * self.feature_dim)  # [cluster_nodes, horizon * feature_dim]
        
        # 通过各自的 MLP 处理 X 和 Y
        X = self.mlp_X(X)  # [cluster_nodes, hid_dim]
        Y = self.mlp_Y(Y)  # [cluster_nodes, hid_dim]
        
        # 拼接 X 和 Y 的输出
        output = torch.cat([X, Y], dim=-1)  # [cluster_nodes, hid_dim * 2]
        
        # 通过输出的 MLP 生成精炼后的 Y
        refined_Y = self.mlp_out(output)  # [cluster_nodes, horizon * feature_dim]
        
        # 将 refined_Y 从 [cluster_nodes, horizon * feature_dim] 转换为 [horizon, cluster_nodes, feature_dim]
        refined_Y = refined_Y.reshape(-1, self.horizon, self.feature_dim).permute(1, 0, 2)  # [horizon, cluster_nodes, feature_dim]
        
        return refined_Y
class BackboneGNN(nn.Module):
    def __init__(self, lookback, horizon, num_layers, gnn_type='GCN', feature_dim=1):
        super(BackboneGNN, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.feature_dim = feature_dim
        
        # GNN网络
        self.gnn = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        self.gnn.append(self._get_gnn_layer(lookback * feature_dim, horizon, gnn_type))
        self.batch_norms.append(nn.BatchNorm1d(horizon))

        # 中间层
        for _ in range(num_layers - 2):
            self.gnn.append(self._get_gnn_layer(horizon, horizon, gnn_type))
            self.batch_norms.append(nn.BatchNorm1d(horizon))

        # 最后一层
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
            raise ValueError(f"不支持的GNN类型: {gnn_type}")

    def forward(self, x, edge_index):
        """
        前向传播方法

        参数:
            tspan: 时间跨度，用于生成输出序列，形状为 [horizon]
            x: 输入张量，形状为 [lookback, num_supernodes, feature_dim]
            edge_index: 边索引，形状为 [2, num_edges]
        
        返回:
            out: GNN输出序列，形状为 [horizon, num_supernodes, feature_dim]
        """
        
        # 1. 转换输入张量的维度
        x = x.permute(1, 0, 2)  # [num_supernodes, lookback, feature_dim]
        x = x.reshape(x.shape[0], -1)  # [num_supernodes, lookback * feature_dim]
        
        # 2. 使用GNN生成输出序列
        for i, layer in enumerate(self.gnn):
            x = layer(x, edge_index)
            if i < len(self.gnn) - 1:  # 不对最后一层应用BatchNorm和ReLU
                x = self.batch_norms[i](x)
                x = F.relu(x)
        
        # 3. 调整输出维度以匹配预期的形状
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
        self.k = self.model_args.k * self.args.data.batch_size  # 假设 'k' 在 model_args 中定义
        self.dt = self.model_args.dt
        
        start_t = (self.lookback) * self.dt
        end_t = (self.lookback + self.horizon - 1) * self.dt
        self.tspan = torch.linspace(start_t, end_t, self.horizon)
        # 定义池化层，输入维度为 lookback
        if self.pool_type == 'sag':
            self.pool = SAGPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'topk':
            self.pool = TopKPooling(self.lookback, ratio=self.pool_ratio)
        elif self.pool_type == 'asa':
            self.pool = ASAPooling(self.lookback, ratio=self.pool_ratio)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")

        # 表征网络
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

        # 状态聚合网络
        self.agc_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.ag_hid_dim),
            nn.ReLU(),
            nn.Linear(self.ag_hid_dim, self.feature_dim),
        )
        self.tanh = nn.Tanh()

        # 主干动力学
        self.BackboneGNN = BackboneGNN(self.lookback, self.horizon, self.gnn_layers, self.gnn_type)

        # 精炼层
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
        # 步骤1：读取数据，不修改维度
        x = batch.x  # [num_nodes, lookback]
        edge_index = batch.edge_index  # [2, num_edges]
        cluster_node_indices = batch.cluster_node_indices  # [total_clusters_nodes]
        cluster_ptr = batch.cluster_ptr  # [num_clusters + 1]

        num_nodes = x.size(0)  # 节点数量
        lookback = x.size(1)  # 时间步数
        feature_dim = self.feature_dim  # lookback

        # 步骤2：生成邻接矩阵 adj 从 edge_index
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + adj.t()  # 确保对称性（无向图）

        # 计算标准化拉普拉斯矩阵
        norm_lap = normalized_laplacian(adj)  # [num_nodes, num_nodes]

        # 步骤3：池化操作
        pooled_x = self.pool(x, edge_index)[0]  # pooled_x: [num_supernodes, lookback]
        num_supernodes = pooled_x.size(0)
        # 步骤4：表征网络
        x = x.permute(1, 0).unsqueeze(-1)  # [lookback, num_nodes, feature_dim]
        pooled_x = pooled_x.permute(1, 0).unsqueeze(-1)  # [lookback, num_supernodes, feature_dim]
        node_repr = self.repr_net_x(x)  # [lookback, num_nodes, ag_hid_dim]
        supernode_repr = self.repr_net_super(pooled_x)  # [lookback, num_supernodes, ag_hid_dim]
        
        # 计算节点表示与超节点表示的相似度
        node_repr_flat = node_repr.reshape(num_nodes, lookback * self.ag_hid_dim)          # [num_nodes, lookback * ag_hid_dim]
        supernode_repr_flat = supernode_repr.reshape(num_supernodes, lookback * self.ag_hid_dim)  # [num_supernodes, lookback * ag_hid_dim]
        similarity = torch.matmul(node_repr_flat, supernode_repr_flat.t())  # [num_nodes, num_supernodes]
        assignment_matrix = F.softmax(similarity, dim=1)  # [num_nodes, num_supernodes]

        # 计算超节点邻接矩阵
        adj_mean = adj.mean(dim=0, keepdim=True).expand_as(adj)  # [num_nodes, num_nodes]
        backbone = torch.matmul(assignment_matrix.t(), torch.matmul(adj_mean, assignment_matrix))  # [num_supernodes, num_supernodes]

        # 步骤5：状态聚合
        agc_repr = self.tanh(
            self.agc_mlp(
                torch.matmul(norm_lap, x)
            )
        )  # [lookback, num_nodes, feature_dim]

        # 使用分配矩阵得到超节点嵌入
        supernode_embeddings = torch.matmul(assignment_matrix.t(), agc_repr)  # [lookback, num_supernodes, feature_dim]

        # 步骤6：主干动力学
        # 将邻接矩阵转换为edge_index格式
        edge_index = torch.nonzero(backbone).t().contiguous()
        Y_supernode = self.BackboneGNN(supernode_embeddings, edge_index)  # [horizon, num_supernodes, feature_dim]

        # 步骤7：映射回原始节点
        Y_coarse = torch.matmul(Y_supernode.squeeze(-1), assignment_matrix.t()).unsqueeze(-1)  # [horizon, num_nodes, feature_dim]

        Y_refine = torch.zeros_like(Y_coarse)  # [horizon, num_nodes, feature_dim]
        if isolate:
            Y_coarse = Y_coarse.detach()

        # 使用 Refiner 精炼预测
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
                # 使用对应的 refiner 对簇内的节点进行精炼预测
                refined_output = self.refiners[k](cluster_X, cluster_Y_coarse)  # [horizon, cluster_nodes, feature_dim]
                # 将精炼的预测结果写回 Y_refine
                Y_refine[:, cluster_nodes, :] = refined_output
        
        return Y_refine.permute(1, 0, 2).squeeze(-1), batch.y # , assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
    
    
    # def loss(self, pred, target, assignment_matrix, supernode_adj, orig_adj):
    def loss(self, pred, target):
        # 预测损失（MSE）
        pred_loss = F.mse_loss(pred, target)
        
        # # 重构损失
        # rg_loss, _ = self._rg_loss(Y_supernode, target, assignment_matrix)

        # # One-hot 损失（鼓励每个节点只属于一个超节点）
        # onehot_loss = self._onehot_loss(assignment_matrix)

        # # 均匀分布损失（鼓励超节点大小均匀）
        # uniform_loss = self._uniform_loss(assignment_matrix)

        # # 重构损失（鼓励保持原始图结构）
        # recons_loss = self._recons_loss(assignment_matrix, orig_adj)

        # # 精炼损失
        # refine_loss, _ = self._refine_loss(Y_refine, target)

        # 总损失
        total_loss = (
            pred_loss 
            # + self.lambda_rg * rg_loss +
            # self.lambda_onehot * onehot_loss +
            # self.lambda_uniform * uniform_loss +
            # self.lambda_recons * recons_loss +
            # self.lambda_refine * refine_loss
        )

        return total_loss

    # ... (添加您提供的辅助方法)
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