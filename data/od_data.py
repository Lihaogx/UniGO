import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import random
import h5py
import tqdm

def rolling_matrix(x,window_size=21):
    x = x.flatten()
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()


class Graph_OD_dataset(Dataset):
    def __init__(self, file_path, data_config):
        super(Graph_OD_dataset, self).__init__()
        self.file_path = file_path
        self.data_config = data_config
        self.data = []
        self.load_data()

    def load_data(self):
        if 'synthetic' in self.file_path:
            random.seed(1234)
            print("load dataset...")
            with h5py.File(self.file_path, 'r') as f:
                all_keys = list(f.keys())
                num_data_items = len(all_keys)
                
                # 计算需要抽取的数据数量
                ratio = self.data_config.get('ratio', 1.0)
                if not (0 < ratio <= 1.0):
                    raise ValueError("data_config.ratio 必须在 (0, 1] 之间")
                
                num_sample = int(num_data_items * ratio)
                if num_sample < 1:
                    num_sample = 1  # 至少抽取一个数据项
                
                # 随机抽取索引
                sampled_indices = random.sample(range(num_data_items), num_sample)
                
                for i in tqdm.tqdm(sampled_indices):
                    group = f[all_keys[i]]
                    # 加载 HDF5 数据并转换为 PyTorch 张量
                    x = torch.tensor(group['x'][:], dtype=torch.float)
                    edge_index = torch.tensor(group['edge_index'][:], dtype=torch.long)
                    y = torch.tensor(group['y'][:], dtype=torch.float)
                    cluster_node_indices = torch.tensor(group['cluster_node_indices'][:], dtype=torch.long)
                    cluster_ptr = torch.tensor(group['cluster_ptr'][:], dtype=torch.long)
                    convergence_step = group.attrs['convergence_step']
    
                    # 创建 MyData 对象
                    data = self.MyData(
                        x=x,
                        edge_index=edge_index,
                        y=y,
                        cluster_node_indices=cluster_node_indices,
                        cluster_ptr=cluster_ptr,
                        convergence_step=convergence_step
                    )
    
                    self.data.append(data)
        elif 'real_data' in self.file_path:
            # 处理真实数据的加载
            loaded_data = torch.load(self.file_path)
            # 假设 loaded_data 已经包含 cluster_node_indices 和 cluster_ptr
            data = self.MyData(
                x=loaded_data.x,
                edge_index=loaded_data.edge_index,
                y=loaded_data.y,
                cluster_node_indices=loaded_data.cluster_node_indices,
                cluster_ptr=loaded_data.cluster_ptr
            )
            self.data.append(data)
        else:
            raise ValueError("无效的文件路径。无法确定数据类型（synthetic 或 real）。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    class MyData(Data):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.num_clusters = 3
            # 其他初始化操作（如果需要）

        def __inc__(self, key, value, *args, **kwargs):
            if key == 'cluster_node_indices':
                return self.num_nodes
            elif key == 'cluster_ptr':
                return self.num_nodes
            else:
                return super().__inc__(key, value, *args, **kwargs)