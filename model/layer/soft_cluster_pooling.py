import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftClusterPooling(nn.Module):
    def __init__(self, input_dim, num_nodes, cluster_type, initial_cluster_param=0.01):
        super(SoftClusterPooling, self).__init__()
        
        self.num_nodes = num_nodes
        if cluster_type == 0:
            self.num_clusters_param = nn.Parameter(torch.tensor(initial_cluster_param))
        else:
            self.num_clusters_param = torch.tensor(cluster_type) if isinstance(cluster_type, int) else None
        base = 2000
        self.num_clusters = int(torch.clamp(torch.tensor(initial_cluster_param * base), min=1).item())

        self.assignment = nn.Linear(input_dim, self.num_clusters)
    def forward(self, x):
        # x: [batch_size * num_nodes, input_dim]
        batch_size = x.size(0) // self.num_nodes
        input_dim = x.size(1)
        
        # [batch_size * num_nodes, num_clusters]
        s = F.softmax(self.assignment(x), dim=-1)  
        #  [batch_size, num_nodes, num_clusters]
        s = s.view(batch_size, self.num_nodes, self.num_clusters)
        #  [batch_size, num_nodes, input_dim]
        x = x.view(batch_size, self.num_nodes, input_dim)


        cluster_sums = torch.bmm(s.permute(0, 2, 1), x)  # [batch_size, num_clusters, input_dim]
        cluster_counts = s.sum(dim=1).view(batch_size, self.num_clusters, 1)  # [batch_size, num_clusters, 1]
        cluster_embeddings = cluster_sums / cluster_counts  # [batch_size, num_clusters, input_dim]

        max_cluster_indices = torch.argmax(s, dim=-1)  # [batch_size, num_nodes]
        new_x = cluster_embeddings.gather(1, max_cluster_indices.unsqueeze(-1).expand(-1, -1, input_dim))
        new_x = new_x.view(batch_size * self.num_nodes, input_dim)

        return new_x, s