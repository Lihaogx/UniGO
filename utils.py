import os
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR
from typing import Iterator, Any
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import time

class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.val_start_time = None
        self.train_times = []
        self.val_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        train_time = time.time() - self.train_start_time
        self.train_times.append(train_time)
        print(f"Epoch {trainer.current_epoch} - Training time: {train_time:.2f} seconds")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        val_time = time.time() - self.val_start_time
        self.val_times.append(val_time)
        print(f"Epoch {trainer.current_epoch} - Validation time: {val_time:.2f} seconds")

    def on_test_epoch_start(self, trainer, pl_module):
        self.val_start_time = time.time()

    def on_test_epoch_end(self, trainer, pl_module):
        val_time = time.time() - self.val_start_time
        self.val_times.append(val_time)
        print(f"Test time: {val_time:.2f} seconds")


def save_all_results_to_csv(all_results, result_file):

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    df = pd.DataFrame(all_results)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    mean_row = df[numeric_columns].mean()
    std_row = df[numeric_columns].std()

    mean_series = pd.Series({'run_id': 'mean', 'seed': '', 'current_time': '', 'best_checkpoint_path': ''})
    std_series = pd.Series({'run_id': 'std', 'seed': '', 'current_time': '', 'best_checkpoint_path': ''})
    for col in numeric_columns:
        mean_series[col] = mean_row[col]
        std_series[col] = std_row[col]
    df = pd.concat([df, pd.DataFrame([mean_series]), pd.DataFrame([std_series])], ignore_index=True)
    
    df.to_csv(result_file, index=False)
    
    
def load_config(file_path):
    return OmegaConf.load(file_path)

def add_test_phase_separator(logger, phase_name):
    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_scalar(f'Test Phase - {phase_name}', 1, global_step=0)
        
        
def set_file_paths(args):
    if args.data.task_type in [1, 2]:
        # Construct file path for synthetic data
        base_dir = f"data/synthetic_data/{args.data.name}"
        file_pattern = f"data_size{args.data.size}"
        # Assuming there's only one file matching this pattern in the directory
        for file in os.listdir(base_dir):
            if file.startswith(file_pattern) and file.endswith('.pkl'):
                args.data.file_path_train = os.path.join(base_dir, file)
                break
        if not hasattr(args.data, 'file_path_train'):
            raise FileNotFoundError(f"No file found starting with '{file_pattern}' in {base_dir}")
    elif args.data.task_type == 3:
        if args.model.network_type in ['sinn']:
            base_dir = f"data/real_data/{args.data.name}"
            for file in os.listdir(base_dir):
                if file.startswith("posts_final") and file.endswith('.tsv'):
                    args.data.file_path_train = os.path.join(base_dir, file)
                elif file.startswith("initial_") and file.endswith('.txt'):
                    args.data.file_path_initial = os.path.join(base_dir, file)
        else:
            # Set file paths for real data
            base_dir = f"data/real_data/{args.data.name}"
            args.data.file_path_train = os.path.join(base_dir, "datax0.pt")
            args.data.file_path_test = os.path.join(base_dir, "dataxm.pt")

            # Check if files exist
            if not os.path.exists(args.data.file_path_train):
                raise FileNotFoundError(f"Training file not found: {args.data.file_path_train}")
            if not os.path.exists(args.data.file_path_test):
                raise FileNotFoundError(f"Test file not found: {args.data.file_path_test}")

    return args


def create_optimizer(params: Iterator[Parameter], optim_config: Any) -> Optimizer:
    """Creates an optimizer based on the configuration."""
    params = filter(lambda p: p.requires_grad, params)
    lr = float(optim_config.get('lr', 1e-3))
    weight_decay = float(optim_config.get('weight_decay', 1e-5))

    if optim_config.optimizer.lower() == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_config.optimizer.lower() == 'sgd':
        momentum = optim_config.get('momentum', 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_config.optimizer.lower() == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optim_config.optimizer}' is not supported")

def create_scheduler(optimizer: Optimizer, optim_config: Any) -> Any:
    """Creates a learning rate scheduler based on the configuration."""
    if not optim_config.get('lr_scheduler', False):
        return None  # No scheduler if lr_scheduler is False or not specified

    scheduler_type = optim_config.get('scheduler', 'cos').lower()
    max_epochs = optim_config.get('max_epoch', 200)

    if scheduler_type == 'step':
        step_size = optim_config.get('step_size', 100)
        lr_decay = optim_config.get('lr_decay', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=lr_decay)
    elif scheduler_type == 'multistep':
        steps = optim_config.get('steps', [100, 150])
        lr_decay = optim_config.get('lr_decay', 0.1)
        return MultiStepLR(optimizer, milestones=steps, gamma=lr_decay)
    elif scheduler_type == 'cos':
        return CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        raise ValueError(f"Scheduler '{scheduler_type}' is not supported")

    
def simple_normalize(tensor):
    sum_tensor = tensor.sum(dim=-1, keepdim=True)
    normalized_tensor = tensor / (sum_tensor + 1e-14)  
    return normalized_tensor

