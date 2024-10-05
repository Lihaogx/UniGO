import torch
from torch import nn
from torchmetrics import Metric

class ODMetrics(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wasserstein_distance_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if torch.isnan(preds).any():
            raise ValueError(f"NaN detected in preds")
        if torch.isnan(target).any():
            raise ValueError(f"NaN detected in target")
        num_nodes = preds.size(0)
        horizon = preds.size(1)
        
        # 计算均方误差（MSE）
        mse = nn.functional.mse_loss(preds, target, reduction='mean')
        self.mse_sum += mse * num_nodes  # 累加 MSE，总和
        
        # 计算每个节点的二阶 Wasserstein 距离
        preds_cpu = preds.detach().cpu()
        target_cpu = target.detach().cpu()
        wd_total = 0.0
        
        for i in range(num_nodes):
            pred_distribution = preds_cpu[i]
            target_distribution = target_cpu[i]
            # 对分布进行排序
            pred_sorted, _ = torch.sort(pred_distribution)
            target_sorted, _ = torch.sort(target_distribution)
            # 计算平方差
            diff = pred_sorted - target_sorted
            squared_diff = diff.pow(2)
            
            if torch.isnan(diff).any():
                raise ValueError(f"NaN detected in diff at node {i}")
            if torch.isnan(squared_diff).any():
                raise ValueError(f"NaN detected in squared_diff at node {i}")
            
            # 计算二阶 Wasserstein 距离
            wd_squared = torch.mean(squared_diff)
            wd = torch.sqrt(wd_squared)
            if torch.isnan(wd_squared):
                raise ValueError(f"NaN detected in wd_squared at node {i}")
            if torch.isnan(wd):
                raise ValueError(f"NaN detected in wd at node {i}")
            
            wd_total += wd.item()
        
        average_wd = wd_total / num_nodes
        self.wasserstein_distance_sum += average_wd * num_nodes  # 累加 Wasserstein 距离，总和
        
        # 检查 wasserstein_distance_sum 是否为 NaN
        if torch.isnan(self.wasserstein_distance_sum):
            raise ValueError(f"NaN detected in wasserstein_distance_sum")
        
        self.total_samples += num_nodes

    def compute(self):
        # 检查 total_samples 是否为 0，防止除以零
        if self.total_samples == 0:
            raise ValueError(f"Total samples is zero, cannot compute metrics")
        
        # 返回平均 MSE 和平均 Wasserstein 距离
        avg_mse = self.mse_sum / self.total_samples
        avg_wd = self.wasserstein_distance_sum / self.total_samples
        
        # 检查 avg_wd 是否为 NaN
        if torch.isnan(avg_wd):
            raise ValueError(f"NaN detected in avg_wd")
        
        return {
            'mse': avg_mse,
            'wasserstein_distance': avg_wd
        }
