import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling, EdgePooling, ASAPooling, PANPooling, MemPooling,SAGEConv
import torchdiffeq as ode
from typing import Optional


class Refiner(nn.Module):
    def __init__(self, lookback, horizon, hid_dim, dropout):
        super(Refiner, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.dropout = dropout
        # 处理 X 的 MLP
        self.mlp_X = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        # 处理 Y 的 MLP
        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        # 输出的 MLP
        self.mlp_out = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, horizon),
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
        X = self.mlp_X(X)  # [cluster_nodes, hid_dim]
        Y = self.mlp_Y(Y)  # [cluster_nodes, hid_dim]
        
        # 拼接 X 和 Y 的输出
        output = torch.cat([X, Y], dim=-1)  # [cluster_nodes, hid_dim * 2]
        
        # 通过输出的 MLP 生成精炼后的 Y
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
        for i in range(self.num_layers - 1):
            x = F.relu(self.layers[i](x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class UniGONet_Pooling(nn.Module):
    def __init__(self, args):
        super(UniGONet_Pooling, self).__init__()
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
        self.num_clusters = self.model_args.num_clusters
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
                in_channels=self.ag_hid_dim,
                out_channels=self.ag_hid_dim,
                heads=4,  # 根据需求调整
                num_clusters=self.num_clusters,  # 根据需求调整
                tau=1.0
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
            nn.Linear(self.ag_hid_dim, self.ag_hid_dim),
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

        # 定义 GraphSAGE 层
        self.graphsage = SAGEConv(self.ag_hid_dim, self.ag_hid_dim)
        # 主干动力学


        self.Backbone = GraphSAGEBackbone(self.ode_hid_dim, self.ode_hid_dim, self.horizon, self.num_layers)
        # 精炼层
        self.refiners = nn.ModuleList([
            Refiner(self.lookback, self.horizon, self.sr_hid_dim, self.dropout) 
            for _ in range(self.k)
        ])
        
        self.shaped_refiner = Refiner(self.lookback, self.horizon, self.sr_hid_dim, self.dropout)
        
        # 定义精炼MLP
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
        # 步骤1：读取数据，不修改维度
        x = batch.x  # [num_nodes, lookback]
        edge_index = batch.edge_index  # [2, num_edges]
        cluster_node_indices = batch.cluster_node_indices  # [total_clusters_nodes]
        cluster_ptr = batch.cluster_ptr  # [num_clusters + 1]

        num_nodes = x.size(0)  # 节点数量
        lookback = x.size(1)  # 时间步数
        feature_dim = self.feature_dim  # lookback

        # 步骤2：生成邻接矩阵 adj 从 edge_index
        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + adj.t()  # 确保对称性
        # adj.fill_diagonal_(1)  # 添加自环
        
        
        
        node_repr = self.repr_net_x(x)  # [num_nodes, ag_hid_dim]
        agc_repr = self.graphsage(node_repr, edge_index)
        agc_repr = self.tanh(agc_repr)
        pooled_x, S = self.pool(agc_repr, batch.batch)
        # 将 pooled_x 的前两个维度合并
        batch_size, num_supernode, dim = pooled_x.shape
        pooled_x = pooled_x.reshape(-1, dim)  # [batch_size * num_supernode, dim]
        supernode_repr = self.repr_net_super(pooled_x)
        assignment_matrix = self.reshape_assignment_matrix(S, batch.batch, self.pool.num_clusters, mask=None)  # [num_nodes_total, super_nodes]

        # 计算超节点邻接矩阵
        temp = torch.matmul(adj, assignment_matrix)  # [num_nodes, num_supernodes]
        backbone = torch.matmul(assignment_matrix.transpose(0, 1), temp)  # [num_supernodes, num_supernodes]

        # 步骤5：状态聚合
        # 使用GraphSAGE层进行消息传递
          # [num_nodes, lookback]


        # 使用分配矩阵得到超节点嵌入
        # supernode_embeddings = torch.matmul(assignment_matrix.transpose(0, 1), agc_repr)  # [num_supernodes, ag_hid_dim]

        # 步骤6：主干动力学
        # 根据backbone生成edge_index
        backbone_edge_index = backbone.nonzero().t()
        Y_supernode = self.Backbone(supernode_repr, backbone_edge_index)  # [horizon, num_supernodes]

        # 步骤7：映射回原始节点
        Y_coarse = torch.matmul(assignment_matrix, Y_supernode)  # [horizon, num_nodes]

        if self.refine:
            if self.refine == 'shape':
                Y_refine = self.shaped_refiner(agc_repr, Y_coarse)
            else:
                Y_refine = torch.zeros_like(Y_coarse)  # [horizon, num_nodes]
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
                        cluster_X = agc_repr[cluster_nodes, :]  # [cluster_nodes, lookback]
                        cluster_Y_coarse = Y_coarse[cluster_nodes, :]  # [cluster_nodes, horizon]
                        # 使用对应的 refiner 对簇内的节点进行精炼预测
                        refined_output = self.refiners[k](cluster_X, cluster_Y_coarse)  # [horizon, cluster_nodes]
                        # 将精炼的预测结果写回 Y_refine
                        Y_refine[cluster_nodes, : ] = refined_output
        else:
            Y_refine = Y_coarse
            
            
        Y_refine = self.mlp_refine(Y_refine)
        if self.other_loss:
            return Y_refine, batch.y , assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
        else:
            return Y_refine, batch.y # assignment_matrix, backbone, adj, # Y_supernode , Y_coarse, x, supernode_embeddings
    
        
            
    
    def reshape_assignment_matrix(
        self,
        S: torch.Tensor, 
        batch: torch.Tensor, 
        num_clusters: int, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将 MemPooling 的 assignment_matrix 从 [batch_size, max_num_nodes, num_clusters]
        重塑为 [num_nodes_total, super_nodes]，其中 super_nodes = batch_size * num_clusters。
        
        Args:
            S (torch.Tensor): 形状为 [batch_size, max_num_nodes, num_clusters] 的分配矩阵。
            batch (torch.Tensor): 形状为 [num_nodes_total] 的批次向量，指示每个节点所属的图。
            num_clusters (int): 每个图的簇数。
            mask (torch.Tensor, optional): 形状为 [batch_size, max_num_nodes] 的掩码矩阵，
                                        指示每个节点是否有效。默认情况下，假设所有节点都是有效的。
        
        Returns:
            torch.Tensor: 形状为 [num_nodes_total, super_nodes] 的全局分配矩阵。
        """
        batch_size, max_num_nodes, K = S.size()
        device = S.device
        super_nodes = batch_size * num_clusters
        num_nodes_total = batch.size(0)

        # 如果 mask 未提供，计算每个图的节点数量并生成 mask
        if mask is None:
            # 计算每个图的节点数量
            counts = torch.bincount(batch, minlength=batch_size)  # [batch_size]
            # 初始化 mask
            mask = torch.zeros((batch_size, max_num_nodes), dtype=torch.bool, device=device)
            # 填充 mask
            for b in range(batch_size):
                num_nodes_b = counts[b].item()
                if num_nodes_b > max_num_nodes:
                    raise ValueError(f"Graph {b} has {num_nodes_b} nodes, which exceeds max_num_nodes {max_num_nodes}")
                mask[b, :num_nodes_b] = 1
        else:
            # 如果 mask 已提供，计算 counts
            counts = mask.sum(dim=1)  # [batch_size]

        # 初始化全局分配矩阵
        assignment_matrix = torch.zeros((num_nodes_total, super_nodes), device=device)

        # 遍历每个图，将其分配矩阵 S 分配到全局 assignment_matrix
        for b in range(batch_size):
            num_nodes_b = counts[b].item()
            if num_nodes_b == 0:
                continue
            # 获取图 b 的 S，形状为 [max_num_nodes, num_clusters]
            S_b = S[b]  # [max_num_nodes, num_clusters]
            # 根据 mask 过滤无效节点，形状为 [num_nodes_b, num_clusters]
            S_b_valid = S_b[:num_nodes_b, :]  # [num_nodes_b, num_clusters]
            # 获取图 b 的节点索引，形状为 [num_nodes_b]
            node_indices = (batch == b).nonzero(as_tuple=False).squeeze(1)  # [num_nodes_b]
            # 计算图 b 的 supernode 索引，形状为 [num_clusters]
            supernode_start = b * num_clusters
            supernode_end = (b + 1) * num_clusters
            supernode_indices = torch.arange(supernode_start, supernode_end, device=device)  # [num_clusters]
            # 使用高级索引将 S_b_valid 分配到 assignment_matrix
            # node_indices.unsqueeze(1): [num_nodes_b, 1]
            # supernode_indices.unsqueeze(0): [1, num_clusters]
            # assignment_matrix[node_indices.unsqueeze(1), supernode_indices.unsqueeze(0)]: [num_nodes_b, num_clusters]
            assignment_matrix[node_indices.unsqueeze(1), supernode_indices.unsqueeze(0)] = S_b_valid

        return assignment_matrix
        
    def loss(self, pred, target, *args):
        # 预测损失（MSE）
        pred_loss = F.mse_loss(pred, target)

        # 总损失
        total_loss = pred_loss

        if self.other_loss and len(args) >= 3:
            assignment_matrix, supernode_adj, orig_adj = args[:3]
            
            # 重构损失
            rg_loss, _ = self._rg_loss(args[3], target, assignment_matrix) if len(args) > 3 else (0, None)

            # One-hot 损失（鼓励每个节点只属于一个超节点）
            onehot_loss = self._onehot_loss(assignment_matrix)

            # 均匀分布损失（鼓励超节点大小均匀）
            uniform_loss = self._uniform_loss(assignment_matrix)

            # # 重构损失（鼓励保持原始图结构）
            # recons_loss = self._recons_loss(assignment_matrix, orig_adj)
            if self.refine:
                # 精炼损失
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
    
    