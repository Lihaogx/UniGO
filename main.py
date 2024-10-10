import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml
import torch
import datetime
import argparse
import os
from model.model_interface import MInterface
from data.data_interface import DInterface
from omegaconf import OmegaConf
from utils import set_file_paths, add_test_phase_separator, load_config, save_all_results_to_csv, TimingCallback
import nni

def load_loggers(logger_dir, current_time_str, seed, run_id):
    loggers = [
        TensorBoardLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}'),
        CSVLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}')
    ]
    return loggers

import os
import pytorch_lightning.callbacks as plc

def load_callbacks(ckpt_dir, current_time_str, seed, run_id, type, name):
    ckpt_path = f"{ckpt_dir}/{name}_{type}_{current_time_str}_{seed}_{run_id}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    callbacks = [
        plc.EarlyStopping(
            monitor='val_wasserstein_distance',
            mode='min',
            patience=50,
            min_delta=0.001
        ),
        plc.ModelCheckpoint(
            monitor='val_wasserstein_distance',
            dirpath=ckpt_path,
            filename='best-wd-{epoch:02d}-{val_wasserstein_distance:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        plc.ModelCheckpoint(
            monitor='val_mse',
            dirpath=ckpt_path,
            filename='best-mse-{epoch:02d}-{val_mse:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        TimingCallback()
    ]

    return callbacks


def run_loop_settings(args):
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random seed is reset to the initial cfg.seed value for each run iteration.
    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(args.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [args.seed + x for x in range(num_iterations)]
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(args.run_multiple_splits)
        seeds = [args.seed] * num_iterations
    return run_ids, seeds



def main(args):
    
    params = nni.get_next_parameter()
    args.optim.lr = params.get('lr', args.optim.lr)
    args.model.ag_hid_dim = params.get('hid_dim', args.model.ag_hid_dim)
    args.model.ode_hid_dim = params.get('hid_dim', args.model.ode_hid_dim)
    args.model.sr_hid_dim = params.get('hid_dim', args.model.sr_hid_dim)
    if args.model_type == 'gnn':
        args.model.gnn_type = params.get('gnn_type', args.model.gnn_type)
        args.model.num_layers = params.get('num_layers', args.model.num_layers)
    if args.model_type == "unigo_reduce":
        args.model.pool_ratio = params.get('pool_ratio', args.model.pool_ratio)
    elif args.model_type in ['unigo', 'unigo_sage']:
        args.model.pool_ratio = params.get('pool_ratio', args.model.pool_ratio)
        args.model.dt = params.get('dt', args.model.dt)
        args.model.pool_type = params.get('pool_type', args.model.pool_type)
        args.model.refine = params.get('refine', args.model.refine)
        args.model.other_loss = params.get('other_loss', args.model.other_loss)
        args.model.rg = params.get('rg', args.model.rg)
        args.model.onehot = params.get('onehot', args.model.onehot)
        args.model.uniform = params.get('uniform', args.model.uniform)
        args.model.refine_loss = params.get('refine_loss', args.model.refine_loss)
        args.model.dropout = params.get('dropout', args.model.dropout)
        args.model.method = params.get('method', args.model.method)
    elif args.model_type == 'unigo_gnn':
        args.model.pool_ratio = params.get('pool_ratio', args.model.pool_ratio)
        args.model.pool_type = params.get('pool_type', args.model.pool_type)
        args.model.gnn_type = params.get('gnn_type', args.model.gnn_type)
        args.model.num_layers = params.get('num_layers', args.model.num_layers)
    
    
    pl.seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.set_default_dtype(torch.float32)
    all_results = []

    for run_id, seed in zip(*run_loop_settings(args)):
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        loggers = load_loggers(args.logger_dir, current_time_str, seed, run_id)
        callbacks = load_callbacks(
            args.ckpt_dir, current_time_str, seed, run_id,
            args.data.task_type, args.data.name
        )
        initial_ckpt_path = args.train.pretrained_ckpt
        
        data_module = DInterface(data_config=args.data)
        
        if args.data.task_type == 'pre-training':
            # 初始化模型，不加载预训练的检查点
            model_module = MInterface(args)
            
            # 设置训练参数
            trainer = Trainer(
                callbacks=callbacks, 
                logger=loggers, 
                max_epochs=args.train.max_epochs, 
                min_epochs=args.train.min_epochs, 
                accelerator=args.train.accelerator, 
                devices=args.train.devices, 
                strategy=args.train.strategy, 
                enable_checkpointing=True,
            )

            # 训练模型
            trainer.fit(model_module, datamodule=data_module)

            # 确定用于测试的最佳检查点
            if args.train.monitor == 'val_wasserstein_distance':
                best_checkpoint_path = callbacks[1].best_model_path
            elif args.train.monitor == 'val_mse':
                best_checkpoint_path = callbacks[2].best_model_path
            else:
                raise ValueError("Unsupported monitoring target specified.")

            # 加载最佳模型检查点进行测试
            model_module = MInterface.load_from_checkpoint(
                checkpoint_path=best_checkpoint_path,
                args=args)

            # 测试模型
            test_results = trainer.test(model=model_module, datamodule=data_module)
            
            for result in test_results:
                result_entry = {
                    "run_id": run_id,
                    "seed": seed,
                    "current_time": current_time_str,
                    "best_checkpoint_path": best_checkpoint_path
                }
                result_entry.update(result)
                all_results.append(result_entry)
            nni.report_final_result(trainer.callback_metrics["test_mse"].item())
            # nni.report_final_result(trainer.callback_metrics["test_wasserstein_distance"].item())
            # 保存结果
            result_file = f'{args.result_dir}/{args.data.task_type}_{args.model_type}/{args.data.name}_{args.model.pool_type}/{current_time_str}.csv'
            save_all_results_to_csv(all_results, result_file)
        
        elif args.data.task_type == 'real-data-testing':
            # 从预训练的检查点加载模型
            model_module = MInterface.load_from_checkpoint(
                checkpoint_path=initial_ckpt_path,
                model_config=args.model,
                optim_config=args.optim
            )
            
            # 设置测试参数
            trainer = Trainer(
                callbacks=callbacks, 
                logger=loggers, 
                accelerator=args.train.accelerator, 
                devices=args.train.devices, 
                strategy=args.train.strategy,
                enable_checkpointing=False,
            )
            data_module.setup()
            # 测试模型
            test_results = trainer.test(model=model_module, datamodule=data_module)
            
            for result in test_results:
                result_entry = {
                    "run_id": run_id,
                    "seed": seed,
                    "current_time": current_time_str,
                    "best_checkpoint_path": initial_ckpt_path
                }
                result_entry.update(result)
                all_results.append(result_entry)
                
            nni.report_final_result(trainer.callback_metrics["val_mse"].item())
            # 保存测试结果
            result_file = f'{args.result_dir}/{args.data.task_type}_{args.model_type}/{args.model.layer_type}/test_{args.data.name}_{args.model.layers}_{args.model.order}_{args.model.cluster_type}.csv'
            save_all_results_to_csv(all_results, result_file)
        
        else:
            raise ValueError(f"未知的 task_type：{args.data.task_type}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    config = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    args = load_config(config.config)
    args = set_file_paths(args)
    main(args)
