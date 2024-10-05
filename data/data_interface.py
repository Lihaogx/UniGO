import torch
import pytorch_lightning as pl
from data.od_data import Graph_OD_dataset
from torch_geometric.loader import DataLoader


class DInterface(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.task_type = data_config.task_type  # 预训练或真实数据测试
        self.file_path_train = data_config.file_path_train
        self.file_path_test = data_config.file_path_test
        self.split = data_config.split  # [train_ratio, val_ratio, test_ratio]
        self.batch_size = data_config.batch_size
        self.data_config = data_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.task_type == 'pre-training':
            # 预训练：将数据集拆分为训练、验证和测试集
            full_dataset = Graph_OD_dataset(self.file_path_train, self.data_config)
            total_size = len(full_dataset)
            train_size = int(total_size * self.split[0])
            val_size = int(total_size * self.split[1])
            test_size = total_size - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        elif self.task_type == 'real-data-testing':
            # 真实数据测试：只加载测试数据集
            self.test_dataset = Graph_OD_dataset(self.file_path_test, self.data_config)
            self.train_dataset = None
            self.val_dataset = None
        else:
            raise ValueError(f"未知的 task_type：{self.task_type}")

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return None  # 或者抛出异常，提示没有训练数据集

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return None  # 或者抛出异常，提示没有验证数据集

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return None  # 或者抛出异常，提示没有测试数据集
