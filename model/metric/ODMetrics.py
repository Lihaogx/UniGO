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
        
        # Calculate Mean Squared Error (MSE)
        mse = nn.functional.mse_loss(preds, target, reduction='mean')
        self.mse_sum += mse * num_nodes  # Accumulate MSE, total sum
        
        # Calculate second-order Wasserstein distance for each node
        preds_cpu = preds.detach().cpu()
        target_cpu = target.detach().cpu()
        wd_total = 0.0
        
        for i in range(num_nodes):
            pred_distribution = preds_cpu[i]
            target_distribution = target_cpu[i]
            # Sort the distributions
            pred_sorted, _ = torch.sort(pred_distribution)
            target_sorted, _ = torch.sort(target_distribution)
            # Calculate squared difference
            diff = pred_sorted - target_sorted
            squared_diff = diff.pow(2)
            
            if torch.isnan(diff).any():
                raise ValueError(f"NaN detected in diff at node {i}")
            if torch.isnan(squared_diff).any():
                raise ValueError(f"NaN detected in squared_diff at node {i}")
            
            # Calculate second-order Wasserstein distance
            wd_squared = torch.mean(squared_diff)
            wd = torch.sqrt(wd_squared)
            if torch.isnan(wd_squared):
                raise ValueError(f"NaN detected in wd_squared at node {i}")
            if torch.isnan(wd):
                raise ValueError(f"NaN detected in wd at node {i}")
            
            wd_total += wd.item()
        
        average_wd = wd_total / num_nodes
        self.wasserstein_distance_sum += average_wd * num_nodes  # Accumulate Wasserstein distance, total sum
        
        # Check if wasserstein_distance_sum is NaN
        if torch.isnan(self.wasserstein_distance_sum):
            raise ValueError(f"NaN detected in wasserstein_distance_sum")
        
        self.total_samples += num_nodes

    def compute(self):
        # Check if total_samples is zero to prevent division by zero
        if self.total_samples == 0:
            raise ValueError(f"Total samples is zero, cannot compute metrics")
        
        # Return average MSE and average Wasserstein distance
        avg_mse = self.mse_sum / self.total_samples
        avg_wd = self.wasserstein_distance_sum / self.total_samples
        
        # Check if avg_wd is NaN
        if torch.isnan(avg_wd):
            raise ValueError(f"NaN detected in avg_wd")
        
        return {
            'mse': avg_mse,
            'wasserstein_distance': avg_wd
        }
